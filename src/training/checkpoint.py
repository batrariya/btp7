from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(path: Path, model, optimizer, epoch: int, group_dro, config: dict, metadata: dict,
                    adversary=None, adversary_optimizer=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "group_dro": group_dro.state_dict() if group_dro else None, "config": config,
                "dataset_metadata": metadata,
                "adversary": adversary.state_dict() if adversary is not None else None,
                "adversary_optimizer": adversary_optimizer.state_dict() if adversary_optimizer is not None else None}, path)


def load_checkpoint(path: str | Path, model, optimizer=None, group_dro=None, device="cpu", adversary=None, adversary_optimizer=None) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if group_dro is not None and state.get("group_dro"):
        group_dro.load_state_dict(state["group_dro"])
    if adversary is not None and state.get("adversary"):
        adversary.load_state_dict(state["adversary"])
    if adversary_optimizer is not None and state.get("adversary_optimizer"):
        adversary_optimizer.load_state_dict(state["adversary_optimizer"])
    return state
