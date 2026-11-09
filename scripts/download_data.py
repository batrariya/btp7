from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets.movielens1m import MovieLens1MDataset

if __name__ == "__main__":
    data = MovieLens1MDataset("data").load()
    print(f"Prepared {data.name}: {data.num_users} users, {data.num_items} items, {len(data.train)} train interactions")
