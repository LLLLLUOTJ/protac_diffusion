from __future__ import annotations

from pathlib import Path

from analyze_train_log import parse_log, stage_summary


def test_parse_log_extracts_node_and_edge(tmp_path: Path) -> None:
    log = """[run] training node diffusion
[data] source=data.pt total=10 train=8 val=2 reasons={}
[train] device=cuda batch_size=16 hidden_dim=128 layers=4 timesteps=200
[epoch 001] train_loss=0.5000 val_loss=0.4000
[checkpoint] saved checkpoints/node.pt
[epoch 002] train_loss=0.3000 val_loss=0.2500
[checkpoint] saved checkpoints/node.pt
[run] training edge diffusion
[data] source=data.pt total=10 train=8 val=2 reasons={}
[train] device=cuda batch_size=8 hidden_dim=128 layers=4 timesteps=200
[epoch 001] train_loss=0.7000 val_loss=0.6000
[checkpoint] saved checkpoints/edge.pt
[epoch 002] train_loss=0.6500 val_loss=0.5500
"""
    log_path = tmp_path / "train.log"
    log_path.write_text(log, encoding="utf-8")

    stages = parse_log(str(log_path))

    assert sorted(stages.keys()) == ["edge", "node"]
    assert len(stages["node"].epochs) == 2
    assert stages["node"].epochs[0].checkpoint_saved is True
    assert stages["node"].epochs[1].checkpoint_saved is True
    assert stages["edge"].epochs[0].checkpoint_saved is True
    assert stages["edge"].epochs[1].checkpoint_saved is False


def test_stage_summary_reports_best_epoch(tmp_path: Path) -> None:
    log_text = """[run] training node diffusion
[data] source=data.pt total=10 train=8 val=2 reasons={}
[train] device=cuda batch_size=16 hidden_dim=128 layers=4 timesteps=200
[epoch 001] train_loss=0.5000 val_loss=0.4000
[checkpoint] saved checkpoints/node.pt
[epoch 002] train_loss=0.3200 val_loss=0.2000
[checkpoint] saved checkpoints/node.pt
[epoch 003] train_loss=0.3100 val_loss=0.2200
"""
    path = tmp_path / "train.log"
    path.write_text(log_text, encoding="utf-8")
    stages = parse_log(str(path))
    summary = stage_summary(stages["node"])
    assert summary["best_epoch"] == 2
    assert abs(summary["best_val_loss"] - 0.2) < 1e-8
    assert abs(summary["val_improvement_pct"] - 50.0) < 1e-8
