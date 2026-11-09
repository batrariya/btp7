"""MovieLens-1M adapter with deterministic per-user chronological splits."""
from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .base_dataset import RecommendationData


class MovieLens1MDataset:
    """Loads ML-1M ratings as implicit positive interactions.

    Ratings >= ``rating_threshold`` are positives.  Per-user timestamp ordering is
    used for an 80/10/10 split, guaranteeing no future item enters train edges.
    """

    name = "movielens1m"

    def __init__(self, data_dir: str | Path, rating_threshold: float = 4.0,
                 split: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42,
                 url: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"):
        self.data_dir, self.rating_threshold, self.split, self.seed, self.url = (
            Path(data_dir), rating_threshold, split, seed, url)

    @property
    def processed_path(self) -> Path:
        tag = f"ml1m_thr{self.rating_threshold:g}_split{'-'.join(map(str, self.split))}_seed{self.seed}.npz"
        return self.data_dir / "processed" / tag

    def _download(self) -> Path:
        raw = self.data_dir / "raw"
        target = raw / "ml-1m.zip"
        extracted = raw / "ml-1m"
        if not (extracted / "ratings.dat").exists():
            raw.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                urllib.request.urlretrieve(self.url, target)
            with zipfile.ZipFile(target) as archive:
                archive.extractall(raw)
        return extracted

    def load(self, force_reprocess: bool = False) -> RecommendationData:
        if self.processed_path.exists() and not force_reprocess:
            return self._load_processed()
        root = self._download()
        ratings = pd.read_csv(root / "ratings.dat", sep="::", engine="python",
                              names=["raw_user", "raw_item", "rating", "timestamp"], encoding="latin-1")
        users = pd.read_csv(root / "users.dat", sep="::", engine="python",
                            names=["raw_user", "gender", "age", "occupation", "zip"], encoding="latin-1")
        positives = ratings[ratings.rating >= self.rating_threshold].copy()
        if positives.empty:
            raise ValueError("No positive interactions: lower rating_threshold.")
        positives = positives.merge(users[["raw_user", "gender"]], on="raw_user", how="left")
        return self._process(positives)

    def _process(self, frame: pd.DataFrame) -> RecommendationData:
        # Retain only users with at least three positives: one in every split.
        counts = frame.groupby("raw_user").size()
        frame = frame[frame.raw_user.isin(counts[counts >= 3].index)].copy()
        raw_users = sorted(frame.raw_user.astype(str).unique(), key=lambda x: int(x))
        raw_items = sorted(frame.raw_item.astype(str).unique(), key=lambda x: int(x))
        uid = {value: index for index, value in enumerate(raw_users)}
        iid = {value: index for index, value in enumerate(raw_items)}
        frame["u"] = frame.raw_user.astype(str).map(uid)
        frame["i"] = frame.raw_item.astype(str).map(iid)
        gender_by_raw = frame.groupby("raw_user").gender.first().astype(str).to_dict()
        groups = np.array([gender_by_raw.get(int(raw), "unknown") if gender_by_raw.get(int(raw), "unknown") in {"M", "F"}
                           else "unknown" for raw in raw_users], dtype=object)
        train: list[tuple[int, int]] = []
        val: Dict[int, List[int]] = {}
        test: Dict[int, List[int]] = {}
        # Stable sorting prevents non-determinism for equal timestamps.
        for user, rows in frame.sort_values(["u", "timestamp", "i"], kind="mergesort").groupby("u", sort=True):
            pairs = rows[["u", "i"]].to_numpy(dtype=np.int64)
            n = len(pairs)
            n_train = max(1, int(np.floor(n * self.split[0])))
            n_val = max(1, int(np.floor(n * self.split[1])))
            if n_train + n_val >= n:
                n_train, n_val = n - 2, 1
            train.extend((int(u), int(i)) for u, i in pairs[:n_train])
            val[int(user)] = [int(i) for _, i in pairs[n_train:n_train + n_val]]
            test[int(user)] = [int(i) for _, i in pairs[n_train + n_val:]]
        data = RecommendationData(self.name, len(uid), len(iid), np.asarray(train, dtype=np.int64),
                                  val, test, groups, ["M", "F", "unknown"], uid, iid)
        self._save_processed(data)
        return data

    def _save_processed(self, data: RecommendationData) -> None:
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.processed_path, train=data.train, groups=data.user_groups,
                            validation=np.array(json.dumps(data.validation)), test=np.array(json.dumps(data.test)),
                            user_map=np.array(json.dumps(data.user_id_map)), item_map=np.array(json.dumps(data.item_id_map)))

    def _load_processed(self) -> RecommendationData:
        archive = np.load(self.processed_path, allow_pickle=True)
        convert = lambda s: {int(k): list(map(int, v)) for k, v in json.loads(str(s)).items()}
        user_map = {str(k): int(v) for k, v in json.loads(str(archive["user_map"])).items()}
        item_map = {str(k): int(v) for k, v in json.loads(str(archive["item_map"])).items()}
        return RecommendationData(self.name, len(user_map), len(item_map), archive["train"].astype(np.int64),
                                  convert(archive["validation"]), convert(archive["test"]), archive["groups"],
                                  ["M", "F", "unknown"], user_map, item_map)
