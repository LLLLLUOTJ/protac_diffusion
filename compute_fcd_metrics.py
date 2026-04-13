"""Compute Fréchet ChemNet Distance (FCD) for molecular datasets.

This script is intentionally lightweight and configuration-driven. It compares
each dataset against a configurable reference dataset and writes CSV / JSON
summaries. It supports CSV / JSON / JSONL inputs and the same anchored-linker
normalization used in the fingerprint analysis script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from fcd_torch import FCD
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fingerprint_analysis"
DEFAULT_REFERENCE_DATASET = "train"
DEFAULT_OUTPUT_PREFIX = "fcd_summary"

DATASETS: list[dict[str, Any]] = [
    {
        "name": "train",
        "path": str(PROJECT_ROOT / "outputs" / "weak_anchor_best" / "weak_anchor_dataset.csv"),
        "role": "train",
        "smiles_col": "anchored_linker_smiles",
    },
    {
        "name": "raw_linker",
        "path": str(PROJECT_ROOT / "data" / "csv" / "linker.csv"),
        "role": "raw",
        "smiles_col": "Smiles_R",
    },
    {
        "name": "gen_16d",
        "path": str(PROJECT_ROOT / "outputs" / "server_sync" / "2026-04-07" / "linker_token_eval.all_generations.json"),
        "role": "generated",
        "smiles_col": "generated_anchored_linker_smiles",
    },
    {
        "name": "gen_pad_suffix_ce015",
        "path": str(
            PROJECT_ROOT
            / "outputs"
            / "server_sync"
            / "2026-04-09"
            / "pad_semantics"
            / "linker_token_eval_pad_semantics_all_generations.json"
        ),
        "role": "generated",
        "smiles_col": "generated_anchored_linker_smiles",
    },
    {
        "name": "gen_pad_suffix_ce005",
        "path": str(
            PROJECT_ROOT
            / "outputs"
            / "server_sync"
            / "2026-04-09"
            / "pad_semantics"
            / "linker_token_eval_ce005_all_generations.json"
        ),
        "role": "generated",
        "smiles_col": "generated_anchored_linker_smiles",
    },
    {
        "name": "gen_32d",
        "path": str(
            PROJECT_ROOT
            / "outputs"
            / "server_sync"
            / "2026-04-09"
            / "pad_semantics"
            / "linker_token_eval_32_all_generations.json"
        ),
        "role": "generated",
        "smiles_col": "generated_anchored_linker_smiles",
    },
    {
        "name": "link_invent",
        "path": str(PROJECT_ROOT / "outputs" / "server_sync" / "2026-04-07" / "linkinvent_compare" / "sampling_32x4.csv"),
        "role": "baseline",
        "smiles_col": "Linker",
    },
]

SMILES_FIELD_CANDIDATES = [
    "smiles",
    "full_smiles",
    "generated_smiles",
    "linker_smiles",
    "generated_anchored_linker_smiles",
    "anchored_linker_smiles",
    "generated_full_smiles",
    "full_protac_smiles",
    "Linker",
    "Smiles",
    "Smiles_R",
]
JSON_RECORD_KEYS = ["records", "data", "items", "results", "samples"]


def log(message: str) -> None:
    print(f"[fcd] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    ensure_dir(stem.parent)
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _extract_records_from_json(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in JSON_RECORD_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        list_values = [value for value in payload.values() if isinstance(value, list)]
        if len(list_values) == 1:
            return list_values[0]
    raise ValueError("Unsupported JSON structure: expected list or dict containing a record list")


def _find_column_case_insensitive(columns: list[str], target: str) -> str | None:
    target_lower = target.lower()
    for column in columns:
        if str(column).lower() == target_lower:
            return str(column)
    return None


def infer_smiles_field(records: list[Any], preferred: str | None = None) -> str | None:
    if not records:
        return preferred
    first = records[0]
    if isinstance(first, str):
        return None
    if not isinstance(first, dict):
        raise ValueError("Unsupported record type")

    columns = [str(key) for key in first.keys()]
    if preferred:
        found = _find_column_case_insensitive(columns, preferred)
        if found is not None:
            return found

    for candidate in SMILES_FIELD_CANDIDATES:
        found = _find_column_case_insensitive(columns, candidate)
        if found is not None:
            return found

    smiles_like = [column for column in columns if "smiles" in column.lower()]
    return smiles_like[0] if smiles_like else None


def load_smiles_records(path: Path, preferred_smiles_col: str | None = None) -> tuple[list[str], str | None]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        smiles_field = infer_smiles_field(df.to_dict(orient="records"), preferred=preferred_smiles_col)
        if smiles_field is None:
            raise ValueError(f"Could not detect smiles column in {path}")
        return [str(value) for value in df[smiles_field].tolist()], smiles_field

    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = _extract_records_from_json(payload)
        if not records:
            return [], preferred_smiles_col
        if all(isinstance(record, str) for record in records):
            return [str(record) for record in records], None
        smiles_field = infer_smiles_field(records, preferred=preferred_smiles_col)
        if smiles_field is None:
            raise ValueError(f"Could not detect smiles field in {path}")
        smiles_values = [str(record.get(smiles_field, "")) for record in records if isinstance(record, dict)]
        return smiles_values, smiles_field

    raise ValueError(f"Unsupported file format: {path}")


def normalize_input_smiles(smiles: str, smiles_field: str | None = None) -> str:
    text = str(smiles or "").strip()
    if not text:
        return text
    if smiles_field is not None and smiles_field.lower() == "smiles_r":
        text = text.replace("[R1]", "[*:1]").replace("[R2]", "[*:2]")
        text = text.replace("[R]", "[*]")
    return text


def canonicalize_smiles(smiles: str) -> str | None:
    text = str(smiles or "").strip()
    if not text:
        return None
    try:
        mol = Chem.MolFromSmiles(text, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def load_canonical_smiles(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(dataset_cfg["path"])).expanduser()
    preferred = dataset_cfg.get("smiles_col")
    smiles_list, detected = load_smiles_records(path, preferred_smiles_col=preferred)
    normalized = [normalize_input_smiles(smiles, smiles_field=detected or preferred) for smiles in smiles_list]

    valid_all: list[str] = []
    unique_seen: set[str] = set()
    valid_unique: list[str] = []
    for smiles in normalized:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            continue
        valid_all.append(canonical)
        if canonical not in unique_seen:
            unique_seen.add(canonical)
            valid_unique.append(canonical)

    return {
        "name": dataset_cfg["name"],
        "role": dataset_cfg["role"],
        "path": str(path),
        "smiles_field": detected or preferred,
        "valid_all": valid_all,
        "valid_unique": valid_unique,
        "num_valid_all": len(valid_all),
        "num_valid_unique": len(valid_unique),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FCD against a chosen reference dataset.")
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE_DATASET,
        help="Dataset name to use as FCD reference. Default: train",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for CSV/JSON outputs. Default: fcd_summary",
    )
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)
    metric = FCD(device="cpu", n_jobs=1)

    datasets = [load_canonical_smiles(cfg) for cfg in DATASETS]
    reference = next((dataset for dataset in datasets if dataset["name"] == args.reference), None)
    if reference is None:
        available = ", ".join(dataset["name"] for dataset in datasets)
        raise RuntimeError(f"Reference dataset '{args.reference}' not found. Available: {available}")

    ref_all = reference["valid_all"]
    ref_unique = reference["valid_unique"]
    log(f"reference={reference['name']} valid_all={len(ref_all)} valid_unique={len(ref_unique)}")

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset["name"] == reference["name"]:
            row = {
                "dataset_name": dataset["name"],
                "role": dataset["role"],
                "reference_dataset": reference["name"],
                "num_valid_all": dataset["num_valid_all"],
                "num_valid_unique": dataset["num_valid_unique"],
                "fcd_vs_reference_all": 0.0,
                "fcd_vs_reference_unique": 0.0,
            }
        else:
            log(f"computing dataset={dataset['name']}")
            fcd_all = float(metric(ref=ref_all, gen=dataset["valid_all"])) if dataset["valid_all"] else None
            fcd_unique = float(metric(ref=ref_unique, gen=dataset["valid_unique"])) if dataset["valid_unique"] else None
            row = {
                "dataset_name": dataset["name"],
                "role": dataset["role"],
                "reference_dataset": reference["name"],
                "num_valid_all": dataset["num_valid_all"],
                "num_valid_unique": dataset["num_valid_unique"],
                "fcd_vs_reference_all": fcd_all,
                "fcd_vs_reference_unique": fcd_unique,
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / f"{args.output_prefix}.csv"
    json_path = OUTPUT_DIR / f"{args.output_prefix}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    plot_df = df[df["role"] != "train"].copy()
    colors = []
    for role in plot_df["role"]:
        if role == "generated":
            colors.append("#4C78A8")
        elif role == "baseline":
            colors.append("#E45756")
        elif role == "raw":
            colors.append("#72B7B2")
        else:
            colors.append("#999999")

    reference_label = reference["name"]
    for metric, title_suffix in [
        ("fcd_vs_reference_all", "All Valid Molecules"),
        ("fcd_vs_reference_unique", "Unique Valid Molecules"),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        x = range(len(plot_df))
        ax.bar(list(x), plot_df[metric].astype(float), color=colors, alpha=0.9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(plot_df["dataset_name"], rotation=30, ha="right")
        ax.set_ylabel("FCD")
        ax.set_title(f"FCD vs {reference_label} ({title_suffix})")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        save_figure(fig, OUTPUT_DIR / f"figure7_{args.output_prefix}_{metric}")

    log(f"csv={csv_path}")
    log(f"json={json_path}")


if __name__ == "__main__":
    main()
