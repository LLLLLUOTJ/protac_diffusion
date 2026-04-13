"""Configuration-driven Morgan fingerprint analysis for molecular datasets.

This script compares multiple molecular datasets using a unified RDKit Morgan
fingerprint representation and descriptor set. It is intended for thesis-ready
analysis of train/test/generated/baseline molecule collections.

What it produces:
- dataset-level summary tables
- per-molecule property CSVs
- nearest-neighbor Tanimoto summaries
- property distribution plots
- per-dataset nearest-neighbor similarity plots
- global and subset t-SNE / PCA plots
- comparison bar charts
- README_analysis.md describing inputs and outputs

Edit the DATASETS list and the top-level configuration block before running.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fingerprint_analysis"

# Edit this list for your own analysis. Each dataset needs a name/path/role.
# Optional field: smiles_col to force a specific column/key.
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
    # Example additional entries:
    # {"name": "gen_32d", "path": "/path/to/results_32.json", "role": "generated"},
    # {"name": "gen_pad_suffix_a", "path": "/path/to/a.json", "role": "generated"},
    # {"name": "gen_pad_suffix_b", "path": "/path/to/b.json", "role": "generated"},
]

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

RANDOM_STATE = 42
MAX_EMBED_SAMPLES_PER_DATASET = 1000
TSNE_PERPLEXITY = 30.0
TSNE_N_ITER = 1000
TSNE_INIT = "pca"

INTERNAL_DIVERSITY_MAX_SAMPLES = 1000
PLOT_DPI = 320
HIST_BINS = 32

ROLE_ORDER = {"train": 0, "raw": 1, "test": 2, "generated": 3, "baseline": 4}
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
]
JSON_RECORD_KEYS = ["records", "data", "items", "results", "samples"]

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)


# -----------------------------------------------------------------------------
# SA score loader
# -----------------------------------------------------------------------------


def load_sascorer() -> Any:
    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore

        return sascorer
    except Exception:
        pass

    search_roots: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        search_roots.append(Path(conda_prefix) / "share" / "RDKit" / "Contrib" / "SA_Score")
    rd_base = os.environ.get("RDBASE")
    if rd_base:
        search_roots.append(Path(rd_base) / "Contrib" / "SA_Score")

    for root in search_roots:
        if root.exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            try:
                return importlib.import_module("sascorer")
            except Exception:
                continue

    searched = ", ".join(str(path) for path in search_roots) if search_roots else "<no contrib paths detected>"
    raise RuntimeError(
        "Failed to load RDKit Contrib sascorer.py. "
        f"Searched: {searched}. "
        "Install RDKit with Contrib support or ensure sascorer.py is importable."
    )


SASCORER = load_sascorer()


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class MoleculeEntry:
    row_index: int
    original_smiles: str
    canonical_smiles: str | None
    mol: Chem.Mol | None
    fp_bitvect: Any | None
    fp_array: np.ndarray | None
    parse_success: bool
    heavy_atoms: int | None
    molecular_weight: float | None
    rotatable_bonds: int | None
    qed: float | None
    sa: float | None
    logp: float | None
    tpsa: float | None


@dataclass
class DatasetAnalysis:
    name: str
    role: str
    path: Path
    smiles_field: str | None
    total_samples: int
    parsed_samples: int
    parse_failures: int
    parse_success_rate: float
    entries: list[MoleculeEntry]
    valid_entries: list[MoleculeEntry]
    unique_valid_entries: list[MoleculeEntry]
    canonical_set: set[str]
    summary: dict[str, Any]
    nn_similarities: list[float]


# -----------------------------------------------------------------------------
# Logging / helpers
# -----------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[analyze] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def safe_median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def safe_quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(values, q))


def sanitize_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text.lower())
    return out.strip("_") or "dataset"


def save_figure(fig: plt.Figure, stem: Path) -> None:
    ensure_dir(stem.parent)
    fig.savefig(stem.with_suffix(".png"), dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def array_from_bitvect(bitvect: Any) -> np.ndarray:
    array = np.zeros((MORGAN_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, array)
    return array


def canonicalize_smiles(smiles: str) -> str | None:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_mol(smiles: str) -> Chem.Mol | None:
    text = str(smiles or "").strip()
    if not text:
        return None
    try:
        mol = Chem.MolFromSmiles(text, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def normalize_input_smiles(smiles: str, smiles_field: str | None = None) -> str:
    text = str(smiles or "").strip()
    if not text:
        return text

    # Normalize raw linker anchor notation from linker.csv into the same dummy
    # atom representation used by train/generated anchored linkers.
    if smiles_field is not None and smiles_field.lower() == "smiles_r":
        text = text.replace("[R1]", "[*:1]").replace("[R2]", "[*:2]")
        text = text.replace("[R]", "[*]")
    return text


def compute_molecule_entry(row_index: int, smiles: str) -> MoleculeEntry:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return MoleculeEntry(
            row_index=row_index,
            original_smiles=str(smiles),
            canonical_smiles=None,
            mol=None,
            fp_bitvect=None,
            fp_array=None,
            parse_success=False,
            heavy_atoms=None,
            molecular_weight=None,
            rotatable_bonds=None,
            qed=None,
            sa=None,
            logp=None,
            tpsa=None,
        )

    canonical = Chem.MolToSmiles(mol, canonical=True)
    bitvect = MORGAN_GENERATOR.GetFingerprint(mol)
    fp_array = array_from_bitvect(bitvect)
    return MoleculeEntry(
        row_index=row_index,
        original_smiles=str(smiles),
        canonical_smiles=canonical,
        mol=mol,
        fp_bitvect=bitvect,
        fp_array=fp_array,
        parse_success=True,
        heavy_atoms=int(mol.GetNumHeavyAtoms()),
        molecular_weight=float(Descriptors.MolWt(mol)),
        rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
        qed=float(QED.qed(mol)),
        sa=float(SASCORER.calculateScore(mol)),
        logp=float(Crippen.MolLogP(mol)),
        tpsa=float(Descriptors.TPSA(mol)),
    )


def choose_color_map(names: Sequence[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {name: cmap(idx % 20) for idx, name in enumerate(names)}


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------


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


def _find_column_case_insensitive(columns: Sequence[str], target: str) -> str | None:
    target_lower = target.lower()
    for column in columns:
        if str(column).lower() == target_lower:
            return str(column)
    return None


def infer_smiles_field(records: Sequence[Any], preferred: str | None = None) -> str | None:
    if not records:
        return preferred
    first = records[0]
    if isinstance(first, str):
        return None
    if not isinstance(first, dict):
        raise ValueError("Unsupported record type: expected dict or string")

    columns = [str(key) for key in first.keys()]
    if preferred:
        exact = _find_column_case_insensitive(columns, preferred)
        if exact is not None:
            return exact

    for candidate in SMILES_FIELD_CANDIDATES:
        exact = _find_column_case_insensitive(columns, candidate)
        if exact is not None:
            return exact

    smiles_like = [column for column in columns if "smiles" in column.lower()]
    if smiles_like:
        return smiles_like[0]
    return None


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
        smiles_values: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Unsupported JSON record in {path}: expected dict")
            smiles_values.append(str(record.get(smiles_field, "")))
        return smiles_values, smiles_field

    raise ValueError(f"Unsupported file format: {path}")


# -----------------------------------------------------------------------------
# Dataset analysis
# -----------------------------------------------------------------------------


def build_dataset_summary(entries: Sequence[MoleculeEntry]) -> dict[str, Any]:
    valid_entries = [entry for entry in entries if entry.parse_success]
    unique_entries = unique_valid_entries(valid_entries)
    novelty_placeholder = None
    return {
        "total_samples": len(entries),
        "parsed_samples": len(valid_entries),
        "parse_failures": len(entries) - len(valid_entries),
        "parse_success_rate": float(len(valid_entries) / len(entries)) if entries else 0.0,
        "unique_valid_count": len(unique_entries),
        "uniqueness": float(len(unique_entries) / len(valid_entries)) if valid_entries else 0.0,
        "novel_unique_count": novelty_placeholder,
        "novelty": novelty_placeholder,
        "internal_diversity": None,
        "mean_nn_tanimoto": None,
        "median_nn_tanimoto": None,
        "q05_nn_tanimoto": None,
        "q25_nn_tanimoto": None,
        "q75_nn_tanimoto": None,
        "q95_nn_tanimoto": None,
        "mean_heavy_atoms": safe_mean([float(entry.heavy_atoms) for entry in valid_entries if entry.heavy_atoms is not None]),
        "mean_mw": safe_mean([float(entry.molecular_weight) for entry in valid_entries if entry.molecular_weight is not None]),
        "mean_rotatable_bonds": safe_mean([float(entry.rotatable_bonds) for entry in valid_entries if entry.rotatable_bonds is not None]),
        "mean_qed": safe_mean([float(entry.qed) for entry in valid_entries if entry.qed is not None]),
        "mean_sa": safe_mean([float(entry.sa) for entry in valid_entries if entry.sa is not None]),
        "mean_logp": safe_mean([float(entry.logp) for entry in valid_entries if entry.logp is not None]),
        "mean_tpsa": safe_mean([float(entry.tpsa) for entry in valid_entries if entry.tpsa is not None]),
    }


def unique_valid_entries(entries: Sequence[MoleculeEntry]) -> list[MoleculeEntry]:
    seen: set[str] = set()
    unique_entries: list[MoleculeEntry] = []
    for entry in entries:
        if not entry.parse_success or entry.canonical_smiles is None:
            continue
        if entry.canonical_smiles in seen:
            continue
        seen.add(entry.canonical_smiles)
        unique_entries.append(entry)
    return unique_entries


def analyze_dataset(dataset_cfg: dict[str, Any]) -> DatasetAnalysis:
    name = str(dataset_cfg["name"])
    role = str(dataset_cfg["role"])
    path = Path(str(dataset_cfg["path"])).expanduser()
    smiles_col = dataset_cfg.get("smiles_col")

    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    log(f"loading dataset={name} role={role} path={path}")
    smiles_list, detected_field = load_smiles_records(path, preferred_smiles_col=smiles_col)
    normalized_smiles = [
        normalize_input_smiles(smiles, smiles_field=detected_field or smiles_col) for smiles in smiles_list
    ]
    entries = [compute_molecule_entry(idx, smiles) for idx, smiles in enumerate(normalized_smiles)]
    valid_entries = [entry for entry in entries if entry.parse_success]
    unique_entries = unique_valid_entries(valid_entries)
    canonical_set = {entry.canonical_smiles for entry in unique_entries if entry.canonical_smiles is not None}
    summary = build_dataset_summary(entries)

    return DatasetAnalysis(
        name=name,
        role=role,
        path=path,
        smiles_field=detected_field or smiles_col,
        total_samples=len(entries),
        parsed_samples=summary["parsed_samples"],
        parse_failures=summary["parse_failures"],
        parse_success_rate=summary["parse_success_rate"],
        entries=entries,
        valid_entries=valid_entries,
        unique_valid_entries=unique_entries,
        canonical_set=canonical_set,
        summary=summary,
        nn_similarities=[],
    )


def compute_internal_diversity(entries: Sequence[MoleculeEntry], max_samples: int, random_state: int) -> float | None:
    fps = [entry.fp_bitvect for entry in entries if entry.fp_bitvect is not None]
    if len(fps) < 2:
        return None
    if len(fps) > max_samples:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(len(fps), size=max_samples, replace=False))
        fps = [fps[int(idx)] for idx in indices]

    similarities: list[float] = []
    for idx, fp in enumerate(fps[:-1]):
        similarities.extend(float(sim) for sim in DataStructs.BulkTanimotoSimilarity(fp, fps[idx + 1 :]))
    if not similarities:
        return None
    return float(1.0 - np.mean(similarities))


def compute_nn_similarity(query_entries: Sequence[MoleculeEntry], reference_entries: Sequence[MoleculeEntry]) -> list[float]:
    reference_fps = [entry.fp_bitvect for entry in reference_entries if entry.fp_bitvect is not None]
    if not reference_fps:
        return []

    similarities: list[float] = []
    for entry in query_entries:
        if entry.fp_bitvect is None:
            continue
        similarities.append(float(max(DataStructs.BulkTanimotoSimilarity(entry.fp_bitvect, reference_fps))))
    return similarities


def apply_reference_metrics(datasets: list[DatasetAnalysis], reference: DatasetAnalysis | None) -> None:
    if reference is None:
        return

    reference_unique = reference.unique_valid_entries
    for dataset in datasets:
        dataset.summary["internal_diversity"] = compute_internal_diversity(
            dataset.unique_valid_entries,
            max_samples=INTERNAL_DIVERSITY_MAX_SAMPLES,
            random_state=RANDOM_STATE,
        )

        if dataset.name == reference.name:
            continue

        dataset.nn_similarities = compute_nn_similarity(dataset.valid_entries, reference_unique)
        dataset.summary["mean_nn_tanimoto"] = safe_mean(dataset.nn_similarities)
        dataset.summary["median_nn_tanimoto"] = safe_median(dataset.nn_similarities)
        dataset.summary["q05_nn_tanimoto"] = safe_quantile(dataset.nn_similarities, 0.05)
        dataset.summary["q25_nn_tanimoto"] = safe_quantile(dataset.nn_similarities, 0.25)
        dataset.summary["q75_nn_tanimoto"] = safe_quantile(dataset.nn_similarities, 0.75)
        dataset.summary["q95_nn_tanimoto"] = safe_quantile(dataset.nn_similarities, 0.95)

        unique_canonical = {entry.canonical_smiles for entry in dataset.unique_valid_entries if entry.canonical_smiles}
        novel = sorted(smiles for smiles in unique_canonical if smiles not in reference.canonical_set)
        dataset.summary["novel_unique_count"] = len(novel)
        dataset.summary["novelty"] = float(len(novel) / len(unique_canonical)) if unique_canonical else 0.0


# -----------------------------------------------------------------------------
# Output tables
# -----------------------------------------------------------------------------


def summary_dataframe(datasets: Sequence[DatasetAnalysis]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        row = {
            "dataset_name": dataset.name,
            "role": dataset.role,
            "path": str(dataset.path),
            "smiles_field": dataset.smiles_field,
            "total_samples": dataset.total_samples,
            "parsed_samples": dataset.parsed_samples,
            "parse_failures": dataset.parse_failures,
            "parse_success_rate": dataset.parse_success_rate,
            "unique_valid_count": dataset.summary["unique_valid_count"],
            "uniqueness": dataset.summary["uniqueness"],
            "novel_unique_count": dataset.summary["novel_unique_count"],
            "novelty": dataset.summary["novelty"],
            "internal_diversity": dataset.summary["internal_diversity"],
            "mean_nn_tanimoto": dataset.summary["mean_nn_tanimoto"],
            "median_nn_tanimoto": dataset.summary["median_nn_tanimoto"],
            "q05_nn_tanimoto": dataset.summary["q05_nn_tanimoto"],
            "q25_nn_tanimoto": dataset.summary["q25_nn_tanimoto"],
            "q75_nn_tanimoto": dataset.summary["q75_nn_tanimoto"],
            "q95_nn_tanimoto": dataset.summary["q95_nn_tanimoto"],
            "mean_heavy_atoms": dataset.summary["mean_heavy_atoms"],
            "mean_mw": dataset.summary["mean_mw"],
            "mean_rotatable_bonds": dataset.summary["mean_rotatable_bonds"],
            "mean_qed": dataset.summary["mean_qed"],
            "mean_sa": dataset.summary["mean_sa"],
            "mean_logp": dataset.summary["mean_logp"],
            "mean_tpsa": dataset.summary["mean_tpsa"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["role_order"] = df["role"].map(ROLE_ORDER).fillna(99)
        df = df.sort_values(["role_order", "dataset_name"]).drop(columns=["role_order"])
    return df


def write_markdown_table(df: pd.DataFrame, path: Path, columns: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    if columns is not None:
        df = df.loc[:, list(columns)]

    data = df.copy()
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
        else:
            data[column] = data[column].map(lambda x: "" if pd.isna(x) else str(x))

    widths = {column: max(len(column), *(len(value) for value in data[column].tolist())) for column in data.columns}
    header = "| " + " | ".join(column.ljust(widths[column]) for column in data.columns) + " |"
    sep = "| " + " | ".join("-" * widths[column] for column in data.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[column]).ljust(widths[column]) for column in data.columns) + " |"
        for _, row in data.iterrows()
    ]
    path.write_text("\n".join([header, sep, *rows]) + "\n", encoding="utf-8")


def plot_basic_stats_table(df: pd.DataFrame, output_stem: Path) -> None:
    columns = [
        "dataset_name",
        "total_samples",
        "parsed_samples",
        "parse_success_rate",
        "uniqueness",
        "novelty",
        "internal_diversity",
    ]
    table_df = df.loc[:, columns].copy()
    for column in table_df.columns:
        if column in {"dataset_name"}:
            continue
        table_df[column] = table_df[column].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}" if isinstance(x, float) else str(x))

    fig_height = max(2.6, 0.45 * (len(table_df) + 2))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")
    ax.set_title("Dataset Summary Metrics", fontsize=13, pad=12)
    table = ax.table(
        cellText=table_df.values.tolist(),
        colLabels=table_df.columns.tolist(),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    save_figure(fig, output_stem)


def write_per_molecule_csv(dataset: DatasetAnalysis, output_dir: Path) -> None:
    ensure_dir(output_dir)
    output_path = output_dir / f"per_molecule_properties_{sanitize_name(dataset.name)}.csv"
    fieldnames = [
        "row_index",
        "dataset_name",
        "role",
        "original_smiles",
        "canonical_smiles",
        "parse_success",
        "heavy_atoms",
        "molecular_weight",
        "rotatable_bonds",
        "qed",
        "sa",
        "logp",
        "tpsa",
        "nn_similarity_to_train",
    ]

    nn_lookup: dict[str, float] = {}
    if dataset.nn_similarities:
        valid_entries = [entry for entry in dataset.valid_entries if entry.canonical_smiles is not None]
        for entry, similarity in zip(valid_entries, dataset.nn_similarities, strict=True):
            if entry.canonical_smiles is not None:
                nn_lookup.setdefault(entry.canonical_smiles, similarity)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in dataset.entries:
            writer.writerow(
                {
                    "row_index": entry.row_index,
                    "dataset_name": dataset.name,
                    "role": dataset.role,
                    "original_smiles": entry.original_smiles,
                    "canonical_smiles": entry.canonical_smiles,
                    "parse_success": entry.parse_success,
                    "heavy_atoms": entry.heavy_atoms,
                    "molecular_weight": entry.molecular_weight,
                    "rotatable_bonds": entry.rotatable_bonds,
                    "qed": entry.qed,
                    "sa": entry.sa,
                    "logp": entry.logp,
                    "tpsa": entry.tpsa,
                    "nn_similarity_to_train": None if entry.canonical_smiles is None else nn_lookup.get(entry.canonical_smiles),
                }
            )


def write_nn_summary_csv(datasets: Sequence[DatasetAnalysis], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        if not dataset.nn_similarities:
            continue
        rows.append(
            {
                "dataset_name": dataset.name,
                "role": dataset.role,
                "num_samples": len(dataset.nn_similarities),
                "mean_nn_tanimoto": safe_mean(dataset.nn_similarities),
                "median_nn_tanimoto": safe_median(dataset.nn_similarities),
                "q05_nn_tanimoto": safe_quantile(dataset.nn_similarities, 0.05),
                "q25_nn_tanimoto": safe_quantile(dataset.nn_similarities, 0.25),
                "q75_nn_tanimoto": safe_quantile(dataset.nn_similarities, 0.75),
                "q95_nn_tanimoto": safe_quantile(dataset.nn_similarities, 0.95),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


METRIC_SPECS = [
    ("heavy_atoms", "Heavy Atom Count"),
    ("molecular_weight", "Molecular Weight"),
    ("rotatable_bonds", "Rotatable Bonds"),
    ("qed", "QED"),
    ("sa", "SA Score"),
    ("logp", "logP"),
    ("tpsa", "TPSA"),
]

VIOLIN_METRICS = [
    ("heavy_atoms", "Heavy atoms"),
    ("molecular_weight", "MW"),
    ("rotatable_bonds", "Rotatable bonds"),
    ("qed", "QED"),
    ("sa", "SA"),
    ("logp", "logP"),
    ("tpsa", "TPSA"),
]


def metric_values(entries: Sequence[MoleculeEntry], key: str) -> list[float]:
    values: list[float] = []
    for entry in entries:
        value = getattr(entry, key)
        if value is None:
            continue
        values.append(float(value))
    return values


def plot_property_distribution(
    datasets: Sequence[DatasetAnalysis],
    key: str,
    title: str,
    colors: dict[str, Any],
    output_stem: Path,
) -> None:
    valid_datasets = [(dataset.name, metric_values(dataset.valid_entries, key)) for dataset in datasets]
    valid_datasets = [(name, values) for name, values in valid_datasets if values]
    if len(valid_datasets) < 1:
        return

    combined = [value for _, values in valid_datasets for value in values]
    lo = min(combined)
    hi = max(combined)
    if math.isclose(lo, hi):
        hi = lo + 1.0
    bins = np.linspace(lo, hi, HIST_BINS + 1)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for name, values in valid_datasets:
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            alpha=0.95,
            label=name,
            color=colors[name],
        )

    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, output_stem)


def plot_nn_histogram(dataset: DatasetAnalysis, colors: dict[str, Any], output_stem: Path) -> None:
    if not dataset.nn_similarities:
        return
    bins = np.linspace(0.0, 1.0, HIST_BINS + 1)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.hist(
        dataset.nn_similarities,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.55,
        color=colors[dataset.name],
        edgecolor=colors[dataset.name],
        linewidth=1.3,
    )
    ax.set_title(f"Nearest-Neighbor Tanimoto to Train: {dataset.name}")
    ax.set_xlabel("Nearest-Neighbor Tanimoto")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    save_figure(fig, output_stem)


def sample_entries_for_embedding(
    datasets: Sequence[DatasetAnalysis],
    max_per_dataset: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rng = np.random.default_rng(random_state)
    sampled_rows: list[dict[str, Any]] = []
    fp_arrays: list[np.ndarray] = []

    for dataset in datasets:
        valid_entries = [entry for entry in dataset.valid_entries if entry.fp_array is not None and entry.canonical_smiles is not None]
        if not valid_entries:
            continue
        if len(valid_entries) > max_per_dataset:
            indices = np.sort(rng.choice(len(valid_entries), size=max_per_dataset, replace=False))
            valid_entries = [valid_entries[int(idx)] for idx in indices]

        for entry in valid_entries:
            sampled_rows.append(
                {
                    "dataset_name": dataset.name,
                    "role": dataset.role,
                    "canonical_smiles": entry.canonical_smiles,
                    "original_smiles": entry.original_smiles,
                }
            )
            fp_arrays.append(entry.fp_array.astype(np.float32))

    if not fp_arrays:
        return [], np.empty((0, MORGAN_BITS), dtype=np.float32)
    return sampled_rows, np.stack(fp_arrays, axis=0)


def compute_effective_perplexity(num_samples: int, target: float) -> float:
    if num_samples <= 3:
        return 1.0
    upper = max(1.0, (num_samples - 1) / 3.0)
    return float(min(target, upper))


def compute_embeddings(fp_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if fp_matrix.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(fp_matrix)

    if fp_matrix.shape[0] < 3:
        tsne_coords = pca_coords.copy()
    else:
        perplexity = compute_effective_perplexity(fp_matrix.shape[0], TSNE_PERPLEXITY)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=RANDOM_STATE,
            init=TSNE_INIT,
            learning_rate="auto",
            max_iter=TSNE_N_ITER,
        )
        tsne_coords = tsne.fit_transform(fp_matrix)
    return pca_coords, tsne_coords


def plot_scatter(
    coords_df: pd.DataFrame,
    dataset_names: Sequence[str],
    colors: dict[str, Any],
    x_col: str,
    y_col: str,
    title: str,
    output_stem: Path,
) -> None:
    subset = coords_df[coords_df["dataset_name"].isin(dataset_names)].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for dataset_name in dataset_names:
        block = subset[subset["dataset_name"] == dataset_name]
        if block.empty:
            continue
        ax.scatter(
            block[x_col],
            block[y_col],
            s=14,
            alpha=0.65,
            label=dataset_name,
            color=colors[dataset_name],
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col.upper())
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, output_stem)


def plot_metric_bar_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    colors: dict[str, Any],
    output_stem: Path,
) -> None:
    plot_df = df.copy()
    plot_df = plot_df[plot_df["role"].isin(["generated", "baseline"])]
    plot_df = plot_df[plot_df[metric].notna()]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6.5, 1.2 * len(plot_df)), 5.2))
    x = np.arange(len(plot_df))
    bar_colors = [colors[name] for name in plot_df["dataset_name"]]
    ax.bar(x, plot_df[metric].astype(float), color=bar_colors, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["dataset_name"], rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    save_figure(fig, output_stem)


def dataset_lookup(datasets: Sequence[DatasetAnalysis]) -> dict[str, DatasetAnalysis]:
    return {dataset.name: dataset for dataset in datasets}


def violin_dataset_names(datasets: Sequence[DatasetAnalysis]) -> list[str]:
    preferred = [
        "train",
        "raw_linker",
        "gen_16d",
        "gen_pad_suffix_ce005",
        "gen_32d",
        "link_invent",
    ]
    lookup = dataset_lookup(datasets)
    chosen = [name for name in preferred if name in lookup]
    if len(chosen) >= 3:
        return chosen
    return [dataset.name for dataset in datasets if dataset.valid_entries]


def draw_violin_panel(
    ax: plt.Axes,
    datasets: Sequence[DatasetAnalysis],
    dataset_names: Sequence[str],
    metric_key: str,
    metric_label: str,
    colors: dict[str, Any],
) -> None:
    lookup = dataset_lookup(datasets)
    names = [name for name in dataset_names if name in lookup]
    values = [metric_values(lookup[name].valid_entries, metric_key) for name in names]
    names, values = zip(*[(name, vals) for name, vals in zip(names, values) if vals], strict=False) if any(values) else ([], [])
    if not names:
        ax.axis("off")
        return

    positions = np.arange(1, len(names) + 1)
    violin = ax.violinplot(values, positions=positions, widths=0.82, showmeans=False, showmedians=True, showextrema=False)
    for body, name in zip(violin["bodies"], names, strict=True):
        body.set_facecolor(colors[name])
        body.set_edgecolor(colors[name])
        body.set_alpha(0.55)
        body.set_linewidth(0.8)
    if "cmedians" in violin:
        violin["cmedians"].set_color("#333333")
        violin["cmedians"].set_linewidth(1.0)

    for pos, vals, name in zip(positions, values, names, strict=True):
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        ax.vlines(pos, q1, q3, color="#333333", linewidth=1.0, alpha=0.75)
        ax.scatter([pos], [np.mean(vals)], marker="o", s=16, color="#333333", zorder=3)

    ax.set_title(metric_label, fontsize=11)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.55)
    ax.tick_params(axis="y", labelsize=8)


def plot_property_violin_panel(
    datasets: Sequence[DatasetAnalysis],
    colors: dict[str, Any],
    output_stem: Path,
) -> None:
    names = violin_dataset_names(datasets)
    if len(names) < 2:
        return

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 9.2))
    axes_flat = axes.flatten()

    for idx, (metric_key, metric_label) in enumerate(VIOLIN_METRICS):
        draw_violin_panel(
            axes_flat[idx],
            datasets=datasets,
            dataset_names=names,
            metric_key=metric_key,
            metric_label=metric_label,
            colors=colors,
        )
        axes_flat[idx].text(
            -0.18,
            1.04,
            chr(ord("a") + idx),
            transform=axes_flat[idx].transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
        )

    axes_flat[-1].axis("off")
    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=9, markerfacecolor=colors[name], markeredgecolor=colors[name], alpha=0.7, label=name)
        for name in names
    ]
    axes_flat[-1].legend(handles=legend_handles, loc="center", frameon=False, fontsize=9)

    fig.suptitle("Property Distributions Across Reference and Generated Linkers", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_stem)


# -----------------------------------------------------------------------------
# Comparison set helpers
# -----------------------------------------------------------------------------


def dataset_names_by_keyword(datasets: Sequence[DatasetAnalysis], keywords: Sequence[str]) -> list[str]:
    out: list[str] = []
    for dataset in datasets:
        lowered = dataset.name.lower()
        if any(keyword in lowered for keyword in keywords):
            out.append(dataset.name)
    return out


def choose_best_generated(summary_df: pd.DataFrame) -> str | None:
    generated = summary_df[summary_df["role"] == "generated"].copy()
    if generated.empty:
        return None

    explicit = generated[generated["dataset_name"].str.contains("best", case=False, regex=False)]
    if not explicit.empty:
        return str(explicit.iloc[0]["dataset_name"])

    if generated["mean_nn_tanimoto"].notna().any():
        generated = generated.sort_values(["mean_nn_tanimoto", "parse_success_rate"], ascending=[False, False])
        return str(generated.iloc[0]["dataset_name"])

    generated = generated.sort_values(["parse_success_rate", "uniqueness"], ascending=[False, False])
    return str(generated.iloc[0]["dataset_name"])


def build_special_comparison_sets(datasets: Sequence[DatasetAnalysis], summary_df: pd.DataFrame) -> dict[str, list[str]]:
    names = [dataset.name for dataset in datasets]
    train_names = [dataset.name for dataset in datasets if dataset.role == "train"]
    raw_names = [dataset.name for dataset in datasets if dataset.role == "raw"]
    baseline_names = [dataset.name for dataset in datasets if dataset.role == "baseline"]
    generated_names = [dataset.name for dataset in datasets if dataset.role == "generated"]

    sets: dict[str, list[str]] = {}
    names_16_32 = train_names + [name for name in names if "16d" in name.lower() or "32d" in name.lower()]
    if len(set(names_16_32)) >= 3:
        sets["train_vs_16d_vs_32d"] = list(dict.fromkeys(names_16_32))

    pad_names = train_names + dataset_names_by_keyword(datasets, ["pad", "suffix"])
    if len(set(pad_names)) >= 3:
        sets["train_vs_pad_suffix_versions"] = list(dict.fromkeys(pad_names))

    best_model = choose_best_generated(summary_df)
    if train_names and baseline_names and best_model is not None:
        sets["train_vs_baseline_vs_best_model"] = list(dict.fromkeys(train_names + baseline_names + [best_model]))

    if train_names and baseline_names:
        highlight_generated = []
        for name in generated_names:
            lowered = name.lower()
            if any(keyword in lowered for keyword in ["16d", "32d", "pad", "suffix"]):
                highlight_generated.append(name)
        for name in highlight_generated:
            sets[f"train_vs_baseline_vs_{sanitize_name(name)}"] = list(dict.fromkeys(train_names + baseline_names + [name]))

    if train_names and raw_names and best_model is not None:
        sets["train_vs_raw_vs_best_model"] = list(dict.fromkeys(train_names + raw_names + [best_model]))

    if train_names and raw_names:
        highlight_generated = []
        for name in generated_names:
            lowered = name.lower()
            if any(keyword in lowered for keyword in ["16d", "32d", "pad", "suffix"]):
                highlight_generated.append(name)
        for name in highlight_generated:
            sets[f"train_vs_raw_vs_{sanitize_name(name)}"] = list(dict.fromkeys(train_names + raw_names + [name]))
        for name in baseline_names:
            sets[f"train_vs_raw_vs_{sanitize_name(name)}"] = list(dict.fromkeys(train_names + raw_names + [name]))

    return sets


# -----------------------------------------------------------------------------
# README
# -----------------------------------------------------------------------------


def build_readme_text(output_dir: Path, datasets: Sequence[DatasetAnalysis]) -> str:
    dataset_lines = "\n".join(
        f"- `{dataset.name}` ({dataset.role}): `{dataset.path}` using field `{dataset.smiles_field}`"
        for dataset in datasets
    )
    return f"""# README_analysis

## Input format

- Configure datasets at the top of `analyze_fingerprint_distributions.py`
- Each dataset entry needs:
  - `name`
  - `path`
  - `role`
- Optional:
  - `smiles_col`
- Supported file formats:
  - CSV
  - JSON (`list[dict]`, `list[str]`, or `dict` containing a record list under keys like `records`, `data`, `items`, `results`, `samples`)
- Automatic SMILES field detection tries common names such as:
  - `smiles`
  - `full_smiles`
  - `generated_smiles`
  - `linker_smiles`
  - `generated_anchored_linker_smiles`
  - `anchored_linker_smiles`

## Current datasets

{dataset_lines}

## Output files

- `summary_metrics.csv`: dataset-level metrics table
- `summary_metrics.md`: markdown version of the dataset-level metrics
- `figure1_basic_stats_table.png/.pdf`: summary table figure
- `per_molecule_properties_<dataset>.csv`: per-molecule canonical SMILES and descriptor table
- `nn_similarity_summary.csv`: nearest-neighbor Tanimoto summary versus train
- `figure2_property_distribution_<metric>.png/.pdf`: descriptor distribution plots
- `figure2_violin_property_panel.png/.pdf`: thesis-style violin panel for core properties
- `figure3_nn_tanimoto_<dataset>.png/.pdf`: per-dataset nearest-neighbor Tanimoto histograms
- `tsne_coordinates.csv`: global t-SNE coordinates
- `pca_coordinates.csv`: global PCA coordinates
- `figure4_tsne_all.png/.pdf`: global t-SNE plot
- `figure5_pca_all.png/.pdf`: global PCA plot
- `figure4_tsne_<comparison>.png/.pdf`: t-SNE subset comparison plots
- `figure5_pca_<comparison>.png/.pdf`: PCA subset comparison plots
- `figure6_bar_<metric>.png/.pdf`: metric comparison bar plots

## How to run

```bash
cd {PROJECT_ROOT}
conda activate diffusion
python analyze_fingerprint_distributions.py
```

## How to modify datasets

- Edit the `DATASETS` list near the top of the script
- Add or remove entries as needed
- If a dataset uses a nonstandard field name, set `smiles_col`

## Notes

- Morgan fingerprint settings are fixed to `radius=2`, `nBits=2048`
- Novelty is computed using canonical SMILES versus the first dataset with `role='train'`
- Internal diversity uses unique valid molecules and `1 - mean(pairwise Tanimoto)`
- t-SNE / PCA subset plots are filtered from the same global coordinate table for comparability
- Output directory: `{output_dir}`
"""


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    log(f"output_dir={OUTPUT_DIR}")

    if not DATASETS:
        raise RuntimeError("DATASETS is empty. Add at least one dataset configuration entry at the top of the script.")

    datasets = [analyze_dataset(dataset_cfg) for dataset_cfg in DATASETS]
    reference = next((dataset for dataset in datasets if dataset.role == "train"), None)
    if reference is None:
        log("no dataset with role='train' found; novelty and nearest-neighbor metrics will be skipped")
    else:
        log(f"reference train dataset={reference.name}")

    apply_reference_metrics(datasets, reference)
    summary_df = summary_dataframe(datasets)

    summary_csv = OUTPUT_DIR / "summary_metrics.csv"
    summary_md = OUTPUT_DIR / "summary_metrics.md"
    nn_summary_csv = OUTPUT_DIR / "nn_similarity_summary.csv"
    basic_table_stem = OUTPUT_DIR / "figure1_basic_stats_table"
    coords_tsne_csv = OUTPUT_DIR / "tsne_coordinates.csv"
    coords_pca_csv = OUTPUT_DIR / "pca_coordinates.csv"

    summary_df.to_csv(summary_csv, index=False)
    write_markdown_table(
        summary_df,
        summary_md,
        columns=[
            "dataset_name",
            "total_samples",
            "parsed_samples",
            "parse_success_rate",
            "uniqueness",
            "novelty",
            "internal_diversity",
            "mean_nn_tanimoto",
        ],
    )
    plot_basic_stats_table(summary_df, basic_table_stem)
    write_nn_summary_csv(datasets, nn_summary_csv)

    for dataset in datasets:
        write_per_molecule_csv(dataset, OUTPUT_DIR)

    colors = choose_color_map(summary_df["dataset_name"].tolist())

    log("plotting property distributions")
    for key, title in METRIC_SPECS:
        plot_property_distribution(
            datasets,
            key=key,
            title=title,
            colors=colors,
            output_stem=OUTPUT_DIR / f"figure2_property_distribution_{sanitize_name(key)}",
        )
    plot_property_violin_panel(
        datasets,
        colors=colors,
        output_stem=OUTPUT_DIR / "figure2_violin_property_panel",
    )

    log("plotting nearest-neighbor similarity histograms")
    for dataset in datasets:
        if dataset.role == "train":
            continue
        plot_nn_histogram(
            dataset,
            colors=colors,
            output_stem=OUTPUT_DIR / f"figure3_nn_tanimoto_{sanitize_name(dataset.name)}",
        )

    log("sampling fingerprints for PCA/t-SNE")
    sampled_rows, fp_matrix = sample_entries_for_embedding(
        datasets,
        max_per_dataset=MAX_EMBED_SAMPLES_PER_DATASET,
        random_state=RANDOM_STATE,
    )
    pca_coords, tsne_coords = compute_embeddings(fp_matrix)

    coords_df = pd.DataFrame(sampled_rows)
    if not coords_df.empty:
        coords_df["pca_x"] = pca_coords[:, 0]
        coords_df["pca_y"] = pca_coords[:, 1]
        coords_df["tsne_x"] = tsne_coords[:, 0]
        coords_df["tsne_y"] = tsne_coords[:, 1]
        coords_df.to_csv(coords_pca_csv, index=False)
        coords_df.to_csv(coords_tsne_csv, index=False)

        plot_scatter(
            coords_df,
            dataset_names=summary_df["dataset_name"].tolist(),
            colors=colors,
            x_col="tsne_x",
            y_col="tsne_y",
            title="Morgan Fingerprint t-SNE",
            output_stem=OUTPUT_DIR / "figure4_tsne_all",
        )
        plot_scatter(
            coords_df,
            dataset_names=summary_df["dataset_name"].tolist(),
            colors=colors,
            x_col="pca_x",
            y_col="pca_y",
            title="Morgan Fingerprint PCA",
            output_stem=OUTPUT_DIR / "figure5_pca_all",
        )

        special_sets = build_special_comparison_sets(datasets, summary_df)
        for label, dataset_names in special_sets.items():
            plot_scatter(
                coords_df,
                dataset_names=dataset_names,
                colors=colors,
                x_col="tsne_x",
                y_col="tsne_y",
                title=f"t-SNE: {label.replace('_', ' ')}",
                output_stem=OUTPUT_DIR / f"figure4_tsne_{sanitize_name(label)}",
            )
            plot_scatter(
                coords_df,
                dataset_names=dataset_names,
                colors=colors,
                x_col="pca_x",
                y_col="pca_y",
                title=f"PCA: {label.replace('_', ' ')}",
                output_stem=OUTPUT_DIR / f"figure5_pca_{sanitize_name(label)}",
            )
    else:
        log("no valid fingerprints available for PCA/t-SNE")

    log("plotting comparison bar charts")
    bar_metrics = [
        ("parse_success_rate", "Parse Success Rate"),
        ("uniqueness", "Uniqueness"),
        ("novelty", "Novelty"),
        ("internal_diversity", "Internal Diversity"),
        ("mean_nn_tanimoto", "Mean Nearest-Neighbor Tanimoto"),
    ]
    for metric, title in bar_metrics:
        plot_metric_bar_chart(
            summary_df,
            metric=metric,
            title=title,
            colors=colors,
            output_stem=OUTPUT_DIR / f"figure6_bar_{sanitize_name(metric)}",
        )

    readme_path = OUTPUT_DIR / "README_analysis.md"
    readme_path.write_text(build_readme_text(OUTPUT_DIR, datasets), encoding="utf-8")

    log(f"summary_csv={summary_csv}")
    log(f"summary_md={summary_md}")
    log(f"nn_summary_csv={nn_summary_csv}")
    log(f"tsne_coordinates={coords_tsne_csv}")
    log(f"pca_coordinates={coords_pca_csv}")
    log(f"readme={readme_path}")


if __name__ == "__main__":
    main()
