from __future__ import annotations

import json
from pathlib import Path

from compare_generation_baselines import _parse_linkinvent_log, build_summary


def test_parse_linkinvent_log_extracts_requested_and_invalid(tmp_path: Path) -> None:
    log_path = tmp_path / "sampling.log"
    log_path.write_text(
        "x\n16:45:42 <INFO> Sampling 128 SMILES from model /tmp/linkinvent.prior\n"
        "16:45:43 <INFO> Removed 5 invalid SMILES\n",
        encoding="utf-8",
    )
    parsed = _parse_linkinvent_log(log_path)
    assert parsed["requested"] == 128
    assert parsed["removed_invalid"] == 5


def test_build_summary_uses_expected_metric_fields(tmp_path: Path) -> None:
    token_linker = tmp_path / "token_linker.json"
    token_dist = tmp_path / "token_dist.json"
    token_sampling = tmp_path / "token_sampling.json"
    link_linker = tmp_path / "link_linker.json"
    link_dist = tmp_path / "link_dist.json"
    link_log = tmp_path / "link.log"
    token_feas = tmp_path / "token_feas.json"
    link_feas = tmp_path / "link_feas.json"

    token_linker.write_text(
        json.dumps(
            {
                "total_generated": 128,
                "uniqueness": 1.0,
                "novelty": 0.95,
                "mean_qed": 0.42,
                "mean_sa": 4.2,
            }
        ),
        encoding="utf-8",
    )
    token_dist.write_text(
        json.dumps(
            {
                "descriptor_delta": {
                    "length_heavy_atoms": {"mean_delta": -1.5},
                    "molecular_weight": {"mean_delta": -20.0},
                    "rotatable_bonds": {"mean_delta": -0.3},
                    "qed": {"mean_delta": -0.11},
                    "sa": {"mean_delta": 0.6},
                }
            }
        ),
        encoding="utf-8",
    )
    token_sampling.write_text(
        json.dumps(
            {
                "requested": 128,
                "decode_rate": 1.0,
                "assembly_rate": 1.0,
                "unique_anchored": 127,
            }
        ),
        encoding="utf-8",
    )
    link_linker.write_text(
        json.dumps(
            {
                "total_generated": 123,
                "uniqueness": 0.93,
                "novelty": 0.97,
                "mean_qed": 0.68,
                "mean_sa": 3.8,
            }
        ),
        encoding="utf-8",
    )
    link_dist.write_text(
        json.dumps(
            {
                "descriptor_delta": {
                    "length_heavy_atoms": {"mean_delta": -1.2},
                    "molecular_weight": {"mean_delta": -27.0},
                    "rotatable_bonds": {"mean_delta": -6.7},
                    "qed": {"mean_delta": 0.15},
                    "sa": {"mean_delta": 0.2},
                }
            }
        ),
        encoding="utf-8",
    )
    link_log.write_text(
        "16:45:42 <INFO> Sampling 128 SMILES from model x\n16:45:43 <INFO> Removed 5 invalid SMILES\n",
        encoding="utf-8",
    )
    token_feas.write_text(
        json.dumps({"num_samples": 128, "num_borderline": 20, "num_fail": 18, "pass_rate": 0.703125}),
        encoding="utf-8",
    )
    link_feas.write_text(
        json.dumps({"num_samples": 123, "num_borderline": 25, "num_fail": 18, "pass_rate": 0.6504065041}),
        encoding="utf-8",
    )

    class Args:
        token_linker_eval = str(token_linker)
        token_distribution = str(token_dist)
        token_sampling_summary = str(token_sampling)
        linkinvent_linker_eval = str(link_linker)
        linkinvent_distribution = str(link_dist)
        linkinvent_log = str(link_log)
        token_feasibility = str(token_feas)
        linkinvent_feasibility = str(link_feas)

    summary = build_summary(Args())
    assert summary["token"]["requested"] == 128
    assert summary["token"]["decode_rate"] == 1.0
    assert summary["token"]["feasibility_pass_rate"] == 0.703125
    assert summary["linkinvent"]["requested"] == 128
    assert summary["linkinvent"]["output_count"] == 123
    assert summary["linkinvent"]["removed_invalid"] == 5
    assert summary["linkinvent"]["feasibility_borderline_rate"] == 25 / 123
