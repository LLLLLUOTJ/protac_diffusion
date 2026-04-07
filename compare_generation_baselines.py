from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare token and Link-INVENT baseline generation outputs.")
    parser.add_argument("--token-linker-eval", type=str, required=True)
    parser.add_argument("--token-distribution", type=str, required=True)
    parser.add_argument("--token-sampling-summary", type=str, required=True)
    parser.add_argument("--linkinvent-linker-eval", type=str, required=True)
    parser.add_argument("--linkinvent-distribution", type=str, required=True)
    parser.add_argument("--linkinvent-log", type=str, required=True)
    parser.add_argument("--token-feasibility", type=str, default="")
    parser.add_argument("--linkinvent-feasibility", type=str, default="")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-plot", type=str, required=True)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_linkinvent_log(path: str | Path) -> dict[str, Any]:
    requested = None
    removed_invalid = 0
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        if "Sampling " in line and "SMILES from model" in line:
            try:
                requested = int(line.split("Sampling ", 1)[1].split(" SMILES", 1)[0].strip())
            except Exception:
                requested = None
        if "Removed " in line and "invalid SMILES" in line:
            try:
                removed_invalid = int(line.split("Removed ", 1)[1].split(" invalid", 1)[0].strip())
            except Exception:
                removed_invalid = 0
    return {"requested": requested, "removed_invalid": removed_invalid}


def _maybe_load(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return _load_json(p)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    token_linker = _load_json(args.token_linker_eval)
    token_dist = _load_json(args.token_distribution)
    token_sampling = _load_json(args.token_sampling_summary)

    link_linker = _load_json(args.linkinvent_linker_eval)
    link_dist = _load_json(args.linkinvent_distribution)
    link_log = _parse_linkinvent_log(args.linkinvent_log)

    token_feas = _maybe_load(args.token_feasibility)
    link_feas = _maybe_load(args.linkinvent_feasibility)

    token_requested = int(token_sampling.get("requested", token_linker.get("total_generated", 0)))
    token_output = int(token_linker.get("total_generated", 0))
    link_requested = int(link_log.get("requested") or link_linker.get("total_generated", 0))
    link_output = int(link_linker.get("total_generated", 0))

    summary = {
        "token": {
            "requested": token_requested,
            "output_count": token_output,
            "raw_validity": float(token_output) / float(max(token_requested, 1)),
            "uniqueness": float(token_linker["uniqueness"]),
            "novelty": float(token_linker["novelty"]),
            "mean_qed": float(token_linker["mean_qed"]),
            "mean_sa": float(token_linker["mean_sa"]),
            "decode_rate": float(token_sampling.get("decode_rate", 0.0)),
            "assembly_rate": float(token_sampling.get("assembly_rate", 0.0)),
            "unique_anchored": int(token_sampling.get("unique_anchored", 0)),
            "length_delta_mean": float(token_dist["descriptor_delta"]["length_heavy_atoms"]["mean_delta"]),
            "mw_delta_mean": float(token_dist["descriptor_delta"]["molecular_weight"]["mean_delta"]),
            "rotatable_delta_mean": float(token_dist["descriptor_delta"]["rotatable_bonds"]["mean_delta"]),
            "qed_delta_mean": float(token_dist["descriptor_delta"]["qed"]["mean_delta"]),
            "sa_delta_mean": float(token_dist["descriptor_delta"]["sa"]["mean_delta"]),
        },
        "linkinvent": {
            "requested": link_requested,
            "output_count": link_output,
            "raw_validity": float(link_output) / float(max(link_requested, 1)),
            "uniqueness": float(link_linker["uniqueness"]),
            "novelty": float(link_linker["novelty"]),
            "mean_qed": float(link_linker["mean_qed"]),
            "mean_sa": float(link_linker["mean_sa"]),
            "removed_invalid": int(link_log.get("removed_invalid", 0)),
            "length_delta_mean": float(link_dist["descriptor_delta"]["length_heavy_atoms"]["mean_delta"]),
            "mw_delta_mean": float(link_dist["descriptor_delta"]["molecular_weight"]["mean_delta"]),
            "rotatable_delta_mean": float(link_dist["descriptor_delta"]["rotatable_bonds"]["mean_delta"]),
            "qed_delta_mean": float(link_dist["descriptor_delta"]["qed"]["mean_delta"]),
            "sa_delta_mean": float(link_dist["descriptor_delta"]["sa"]["mean_delta"]),
        },
    }

    if token_feas is not None:
        summary["token"]["feasibility"] = token_feas
        summary["token"]["feasibility_pass_rate"] = float(token_feas.get("pass_rate", 0.0))
        summary["token"]["feasibility_borderline_rate"] = float(token_feas.get("num_borderline", 0)) / float(
            max(int(token_feas.get("num_samples", 0)), 1)
        )
        summary["token"]["feasibility_fail_rate"] = float(token_feas.get("num_fail", 0)) / float(
            max(int(token_feas.get("num_samples", 0)), 1)
        )
    if link_feas is not None:
        summary["linkinvent"]["feasibility"] = link_feas
        summary["linkinvent"]["feasibility_pass_rate"] = float(link_feas.get("pass_rate", 0.0))
        summary["linkinvent"]["feasibility_borderline_rate"] = float(link_feas.get("num_borderline", 0)) / float(
            max(int(link_feas.get("num_samples", 0)), 1)
        )
        summary["linkinvent"]["feasibility_fail_rate"] = float(link_feas.get("num_fail", 0)) / float(
            max(int(link_feas.get("num_samples", 0)), 1)
        )
    return summary


def plot_summary(summary: dict[str, Any], output_path: str | Path) -> None:
    token = summary["token"]
    link = summary["linkinvent"]
    labels = ["Token", "Link-INVENT"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    quality_names = ["raw_validity", "uniqueness", "novelty", "mean_qed"]
    quality_token = [token[name] for name in quality_names]
    quality_link = [link[name] for name in quality_names]
    x = range(len(quality_names))
    width = 0.35
    axes[0, 0].bar([i - width / 2 for i in x], quality_token, width=width, label=labels[0], color="#1f77b4")
    axes[0, 0].bar([i + width / 2 for i in x], quality_link, width=width, label=labels[1], color="#ff7f0e")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(["valid", "uniq", "novel", "QED"])
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Quality Metrics")
    axes[0, 0].legend()

    axes[0, 1].bar(labels, [token["mean_sa"], link["mean_sa"]], color=["#1f77b4", "#ff7f0e"])
    axes[0, 1].set_title("Mean SA")

    if "feasibility_pass_rate" in token and "feasibility_pass_rate" in link:
        feas_names = ["pass", "borderline", "fail"]
        feas_token = [
            token["feasibility_pass_rate"],
            token["feasibility_borderline_rate"],
            token["feasibility_fail_rate"],
        ]
        feas_link = [
            link["feasibility_pass_rate"],
            link["feasibility_borderline_rate"],
            link["feasibility_fail_rate"],
        ]
        x_feas = range(len(feas_names))
        axes[0, 2].bar(
            [i - width / 2 for i in x_feas],
            feas_token,
            width=width,
            label=labels[0],
            color="#1f77b4",
        )
        axes[0, 2].bar(
            [i + width / 2 for i in x_feas],
            feas_link,
            width=width,
            label=labels[1],
            color="#ff7f0e",
        )
        axes[0, 2].set_xticks(list(x_feas))
        axes[0, 2].set_xticklabels(feas_names)
        axes[0, 2].set_ylim(0, 1.05)
        axes[0, 2].set_title("Free-Full Feasibility")
    else:
        axes[0, 2].axis("off")

    delta_names = ["length_delta_mean", "mw_delta_mean", "rotatable_delta_mean", "qed_delta_mean", "sa_delta_mean"]
    delta_token = [token[name] for name in delta_names]
    delta_link = [link[name] for name in delta_names]
    x2 = range(len(delta_names))
    axes[1, 0].bar([i - width / 2 for i in x2], delta_token, width=width, label=labels[0], color="#1f77b4")
    axes[1, 0].bar([i + width / 2 for i in x2], delta_link, width=width, label=labels[1], color="#ff7f0e")
    axes[1, 0].set_xticks(list(x2))
    axes[1, 0].set_xticklabels(["len", "MW", "rot", "QED", "SA"])
    axes[1, 0].set_title("Distribution Shift vs Train")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)

    count_names = ["requested", "output_count"]
    count_token = [token[name] for name in count_names]
    count_link = [link[name] for name in count_names]
    x3 = range(len(count_names))
    axes[1, 1].bar([i - width / 2 for i in x3], count_token, width=width, label=labels[0], color="#1f77b4")
    axes[1, 1].bar([i + width / 2 for i in x3], count_link, width=width, label=labels[1], color="#ff7f0e")
    axes[1, 1].set_xticks(list(x3))
    axes[1, 1].set_xticklabels(["requested", "kept"])
    axes[1, 1].set_title("Sample Counts")

    axes[1, 2].axis("off")
    summary_lines = [
        f"Token decode/assembly: {token.get('decode_rate', 0.0):.3f} / {token.get('assembly_rate', 0.0):.3f}",
        f"Token unique anchored: {token.get('unique_anchored', 0)}",
        f"Link-INVENT removed invalid: {link.get('removed_invalid', 0)}",
        f"Token novelty: {token['novelty']:.3f}",
        f"Link-INVENT novelty: {link['novelty']:.3f}",
    ]
    axes[1, 2].text(0.0, 0.95, "\n".join(summary_lines), va="top", ha="left", fontsize=10)
    axes[1, 2].set_title("Quick Notes")

    fig.suptitle("Token vs Link-INVENT Baseline Comparison", fontsize=14)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    output_json = Path(args.output_json)
    output_plot = Path(args.output_plot)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_summary(summary, output_plot)
    print(f"[done] summary={output_json}")
    print(f"[done] plot={output_plot}")


if __name__ == "__main__":
    main()
