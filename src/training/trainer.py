"""Training orchestration for FairSkew-GCN."""
from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from datasets.base_dataset import RecommendationData
from evaluation.evaluator import Evaluator
from losses.bpr_loss import bpr_per_example
from losses.contrastive_loss import info_nce
from losses.fairness_loss import soft_exposure_losses
from losses.group_dro import GroupDRO
from models.fairskew_gcn import FairSkewGCN
from models.adversarial import EgoSensitiveDiscriminator, ego_item_summary
from .checkpoint import load_checkpoint, save_checkpoint
from utils.logger import setup_logger
from utils.result_formatter import format_table, save_results


class Trainer:
    def __init__(self, model: FairSkewGCN, data: RecommendationData, config: dict, device: torch.device):
        self.model, self.data, self.config, self.device = model, data, config, device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(config["output_dir"]) / f"{config['experiment_name']}-{tag}-seed{config['seed']}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "config.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
        self.logger = setup_logger(self.run_dir / "training.log")
        self.evaluator = Evaluator(data, config["evaluation_ks"], device, config.get("fairness_metric_variant", "divergence"),
                                   config.get("rerank_strength", 0.0), config.get("rerank_pool_size", 100),
                                   config.get("rerank_common_relevance", False),
                                   config.get("rerank_eo_proxy_strength", 0.0))
        labels = {"M": 0, "F": 1}
        group_ids = np.array([labels.get(str(value), -1) for value in data.user_groups], dtype=np.int64)
        self.group_ids = torch.as_tensor(group_ids, device=device)
        self.group_names = [name for name in ("M", "F") if name in data.eligible_groups()]
        self.group_dro = GroupDRO(self.group_names, config["group_dro_eta"], device) if config["enable_group_dro"] else None
        self.name_to_id = {"M": 0, "F": 1}
        self.edges = torch.as_tensor(data.train, dtype=torch.long, device=device)
        self.seen = data.train_items_by_user()
        self.rng = np.random.default_rng(config["seed"])
        self.adversary = None
        self.adversary_optimizer = None
        if config.get("enable_ego_adversary", False):
            self.adversary = EgoSensitiveDiscriminator(config["embedding_dim"], config.get("adversary_hidden_dim")).to(device)
            self.adversary_optimizer = torch.optim.Adam(self.adversary.parameters(), lr=config.get("adversary_learning_rate", 0.001))

    def _sample_negatives(self) -> torch.Tensor:
        """Uniform rejection sampling against each user's *training* positives."""
        users = self.data.train[:, 0]
        neg = self.rng.integers(0, self.data.num_items, size=len(users), dtype=np.int64)
        pending = np.ones(len(users), dtype=bool)
        # ML-1M users are far from exhausting the catalog; this terminates rapidly.
        for _ in range(50):
            bad_positions = np.flatnonzero(pending)
            if not len(bad_positions):
                break
            pending[bad_positions] = np.fromiter(
                (int(neg[index]) in self.seen[int(users[index])] for index in bad_positions), dtype=bool, count=len(bad_positions))
            if pending.any():
                neg[pending] = self.rng.integers(0, self.data.num_items, size=int(pending.sum()), dtype=np.int64)
        if pending.any():
            raise RuntimeError("Could not find negative items; a user may have consumed the full catalog.")
        return torch.as_tensor(neg, device=self.device)

    def _recommendation_loss(self, users: torch.Tensor, item_embeddings: torch.Tensor) -> tuple[torch.Tensor, dict]:
        negatives = self._sample_negatives()
        positive_scores = (users[self.edges[:, 0]] * item_embeddings[self.edges[:, 1]]).sum(-1)
        negative_scores = (users[self.edges[:, 0]] * item_embeddings[negatives]).sum(-1)
        losses = bpr_per_example(positive_scores, negative_scores)
        stats = {"bpr": float(losses.mean().detach())}
        if self.group_dro is None:
            return losses.mean(), stats
        group_losses, active = [], []
        for name in self.group_names:
            mask = self.group_ids[self.edges[:, 0]] == self.name_to_id[name]
            active.append(bool(mask.any()))
            group_losses.append(losses[mask].mean() if mask.any() else losses.new_zeros(()))
        group_losses_t = torch.stack(group_losses)
        active_t = torch.tensor(active, device=self.device, dtype=torch.bool)
        self.group_dro.update(group_losses_t, active_t)
        weighted = (self.group_dro.weights * group_losses_t).sum()
        for name, value, weight in zip(self.group_names, group_losses_t, self.group_dro.weights):
            stats[f"loss_{name}"] = float(value.detach())
            stats[f"weight_{name}"] = float(weight.detach())
        return weighted, stats

    def _adversarial_losses(self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Alternating FairGo-style discriminator/recommender adversarial step."""
        if self.adversary is None or self.adversary_optimizer is None:
            zero = user_embeddings.new_zeros(())
            return zero, zero
        known = self.group_ids >= 0
        if known.sum() < 2:
            zero = user_embeddings.new_zeros(())
            return zero, zero
        ego = ego_item_summary(user_embeddings, item_embeddings, self.edges)
        self.adversary.train()
        discriminator_loss = torch.nn.functional.cross_entropy(self.adversary(user_embeddings.detach()[known], ego.detach()[known]),
                                                               self.group_ids[known])
        self.adversary_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        self.adversary_optimizer.step()
        # Freeze discriminator weights but preserve the gradient from its output
        # to LightGCN representations: the recommender maximizes this CE loss.
        for parameter in self.adversary.parameters():
            parameter.requires_grad_(False)
        fool_loss = torch.nn.functional.cross_entropy(self.adversary(user_embeddings[known], ego[known]), self.group_ids[known])
        for parameter in self.adversary.parameters():
            parameter.requires_grad_(True)
        return discriminator_loss.detach(), fool_loss

    def train(self) -> tuple[dict, Path]:
        start_epoch = 1
        if self.config.get("resume"):
            state = load_checkpoint(self.config["resume"], self.model, self.optimizer, self.group_dro, self.device,
                                    self.adversary, self.adversary_optimizer)
            start_epoch = int(state["epoch"]) + 1
            self.logger.info("Resumed from epoch %s", state["epoch"])
        best_value = -float("inf")
        for epoch in range(start_epoch, self.config["epochs"] + 1):
            tick = time.perf_counter()
            self.model.train()
            original_users, original_items, view_one, view_two = self.model.training_embeddings(self.config["enable_augmentation"])
            group_loss, stats = self._recommendation_loss(original_users, original_items)
            if self.config["enable_contrastive"]:
                size = min(self.config["contrastive_batch_size"], self.data.num_users)
                sample = torch.randperm(self.data.num_users, device=self.device)[:size]
                cl_loss = info_nce(view_one[sample], view_two[sample], self.config["contrastive_temperature"])
            else:
                cl_loss = group_loss.new_zeros(())
            if self.config.get("enable_exposure_fairness", False):
                fairness_size = min(self.config.get("exposure_batch_size", 512), self.data.num_users)
                fair_users = torch.randperm(self.data.num_users, device=self.device)[:fairness_size]
                dp_loss, eo_loss = soft_exposure_losses(original_users, original_items, fair_users, self.group_ids,
                                                         self.edges, self.data.num_items,
                                                         self.config.get("exposure_temperature", 1.0))
            else:
                dp_loss = eo_loss = group_loss.new_zeros(())
            discriminator_loss, adversarial_loss = self._adversarial_losses(original_users, original_items)
            warmup_epochs = max(1, int(self.config.get("fairness_warmup_epochs", 1)))
            fairness_scale = min(1.0, epoch / warmup_epochs)
            reg_loss = (self.model.user_embedding.weight.square().sum() + self.model.item_embedding.weight.square().sum())
            total = (group_loss + self.config["contrastive_coefficient"] * cl_loss +
                     fairness_scale * self.config.get("exposure_dp_coefficient", 0.0) * dp_loss +
                     fairness_scale * self.config.get("exposure_eo_coefficient", 0.0) * eo_loss +
                     - fairness_scale * self.config.get("adversarial_coefficient", 0.0) * adversarial_loss +
                     self.config["regularization"] * reg_loss)
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            stats.update(total=float(total.detach()), contrastive=float(cl_loss.detach()), soft_js_dp=float(dp_loss.detach()),
                         soft_js_eo=float(eo_loss.detach()), fairness_scale=fairness_scale, regularization=float(reg_loss.detach()),
                         discriminator=float(discriminator_loss), adversarial=float(adversarial_loss.detach()),
                         seconds=time.perf_counter() - tick, lr=self.optimizer.param_groups[0]["lr"])
            line = " ".join(f"{key}={value:.5f}" for key, value in stats.items())
            self.logger.info("epoch=%d %s", epoch, line)
            if epoch % self.config["validation_interval"] == 0 or epoch == self.config["epochs"]:
                self.model.eval()
                validation = self.evaluator.evaluate(self.model, self.model.full_adjacency(), "validation")
                score = validation["NDCG"][max(self.config["evaluation_ks"])]
                self.logger.info("epoch=%d validation NDCG@%d=%.6f Recall@%d=%.6f", epoch, max(self.config["evaluation_ks"]), score,
                                 max(self.config["evaluation_ks"]), validation["Recall"][max(self.config["evaluation_ks"])])
                if score > best_value:
                    best_value = score
                    save_checkpoint(self.run_dir / "checkpoints" / "best.pt", self.model, self.optimizer, epoch, self.group_dro,
                                    self.config, {"dataset": self.data.name, "num_users": self.data.num_users, "num_items": self.data.num_items},
                                    self.adversary, self.adversary_optimizer)
            if epoch % self.config["checkpoint_interval"] == 0:
                save_checkpoint(self.run_dir / "checkpoints" / f"epoch-{epoch}.pt", self.model, self.optimizer, epoch, self.group_dro,
                                self.config, {"dataset": self.data.name, "num_users": self.data.num_users, "num_items": self.data.num_items},
                                self.adversary, self.adversary_optimizer)
        save_checkpoint(self.run_dir / "checkpoints" / "final.pt", self.model, self.optimizer, self.config["epochs"], self.group_dro,
                        self.config, {"dataset": self.data.name, "num_users": self.data.num_users, "num_items": self.data.num_items},
                        self.adversary, self.adversary_optimizer)
        # Tuning jobs intentionally end here: their test split is never ranked,
        # inspected, or serialized, so hyperparameter selection stays validation-only.
        if not self.config.get("final_test_evaluation", True):
            self.logger.info("Validation-only tuning run completed; final test evaluation intentionally skipped.")
            return {}, self.run_dir
        self.model.eval()
        metrics = self.evaluator.evaluate(self.model, self.model.full_adjacency(), "test")
        metadata = {"dataset": self.data.name, "model": self.config["experiment_name"], "epoch": self.config["epochs"],
                    "seed": self.config["seed"], "config": self.config}
        save_results(self.run_dir / "results", metrics, metadata, self.config["evaluation_ks"])
        table = format_table("FairSkew-GCN" if self.config["enable_augmentation"] else "LightGCN", self.config["epochs"], metrics, self.config["evaluation_ks"])
        print(table)
        self.logger.info("Final test results saved in %s", self.run_dir / "results")
        return metrics, self.run_dir
