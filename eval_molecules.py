"""Lightweight SMILES evaluation CLI for generated molecules.

Computes:
- validity
- uniqueness
- novelty (optional, requires training/reference file)
- mean QED
- mean SA score

Example:
    python eval_molecules.py \
      --generated generated.csv \
      --smiles-col smiles \
      --output results.json

    python eval_molecules.py \
      --generated generated.json \
      --train train.csv \
      --smiles-col generated_smiles \
      --train-smiles-col smiles \
      --output results.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def load_sascorer() -> Any:
    """Load RDKit Contrib sascorer with a small fallback chain."""

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

    added_paths: list[str] = []
    for root in search_roots:
        if root.exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
                added_paths.append(root_str)
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


def canonicalize_smiles(smiles: str) -> str | None:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _load_json_smiles(path: Path, smiles_col: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(item, str) for item in payload):
            return [str(item) for item in payload]
        if all(isinstance(item, dict) for item in payload):
            smiles_list: list[str] = []
            for item in payload:
                if smiles_col not in item:
                    raise ValueError(f"JSON object is missing required smiles field: {smiles_col}")
                smiles_list.append(str(item[smiles_col]))
            return smiles_list
    raise ValueError(f"Unsupported JSON structure in {path}: expected list[str] or list[object]")


def load_smiles_file(path: str | Path, smiles_col: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"input file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
        if smiles_col not in df.columns:
            raise ValueError(f"CSV column not found in {file_path}: {smiles_col}")
        return [str(value) for value in df[smiles_col].tolist()]
    if suffix == ".json":
        return _load_json_smiles(file_path, smiles_col=smiles_col)
    raise ValueError(f"Unsupported input format for {file_path}: expected .csv or .json")


def compute_validity(smiles_list: Sequence[str]) -> dict[str, Any]:
    valid_mols: list[Chem.Mol] = []
    valid_canonical_smiles: list[str] = []
    for smiles in smiles_list:
        mol = smiles_to_mol(smiles)
        if mol is None:
            continue
        valid_mols.append(mol)
        valid_canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))

    total_count = len(smiles_list)
    valid_count = len(valid_mols)
    validity = (valid_count / total_count) if total_count > 0 else 0.0
    return {
        "total_count": total_count,
        "valid_count": valid_count,
        "validity": float(validity),
        "valid_mols": valid_mols,
        "valid_canonical_smiles": valid_canonical_smiles,
    }


def compute_uniqueness(valid_canonical_smiles: Sequence[str]) -> dict[str, Any]:
    valid_count = len(valid_canonical_smiles)
    unique_valid_smiles = sorted(set(valid_canonical_smiles))
    unique_valid_count = len(unique_valid_smiles)
    uniqueness = (unique_valid_count / valid_count) if valid_count > 0 else 0.0
    return {
        "unique_valid_smiles": unique_valid_smiles,
        "unique_valid_count": unique_valid_count,
        "uniqueness": float(uniqueness),
    }


def compute_novelty(unique_valid_smiles: Sequence[str], train_smiles: Sequence[str]) -> dict[str, Any]:
    train_valid_canonical = {
        canon
        for canon in (canonicalize_smiles(smiles) for smiles in train_smiles)
        if canon is not None
    }
    novel_unique_smiles = sorted(smiles for smiles in unique_valid_smiles if smiles not in train_valid_canonical)
    unique_valid_count = len(unique_valid_smiles)
    novel_unique_count = len(novel_unique_smiles)
    novelty = (novel_unique_count / unique_valid_count) if unique_valid_count > 0 else 0.0
    return {
        "train_valid_unique_count": len(train_valid_canonical),
        "novel_unique_smiles": novel_unique_smiles,
        "novel_unique_count": novel_unique_count,
        "novelty": float(novelty),
    }


def compute_property_scores(valid_mols: Sequence[Chem.Mol]) -> dict[str, Any]:
    if not valid_mols:
        return {"mean_qed": None, "mean_sa": None}

    qed_scores = [float(QED.qed(mol)) for mol in valid_mols]
    sa_scores = [float(SASCORER.calculateScore(mol)) for mol in valid_mols]
    mean_qed = sum(qed_scores) / float(len(qed_scores))
    mean_sa = sum(sa_scores) / float(len(sa_scores))
    return {"mean_qed": float(mean_qed), "mean_sa": float(mean_sa)}


def build_results_dict(
    *,
    generated_path: str,
    train_path: str | None,
    smiles_col: str,
    train_smiles_col: str | None,
    validity_result: dict[str, Any],
    uniqueness_result: dict[str, Any],
    novelty_result: dict[str, Any] | None,
    property_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_path": str(generated_path),
        "train_path": None if train_path is None else str(train_path),
        "smiles_col": str(smiles_col),
        "train_smiles_col": None if train_smiles_col is None else str(train_smiles_col),
        "total_generated": int(validity_result["total_count"]),
        "valid_count": int(validity_result["valid_count"]),
        "validity": float(validity_result["validity"]),
        "unique_valid_count": int(uniqueness_result["unique_valid_count"]),
        "uniqueness": float(uniqueness_result["uniqueness"]),
        "train_valid_unique_count": None if novelty_result is None else int(novelty_result["train_valid_unique_count"]),
        "novel_unique_count": None if novelty_result is None else int(novelty_result["novel_unique_count"]),
        "novelty": None if novelty_result is None else float(novelty_result["novelty"]),
        "mean_qed": property_result["mean_qed"],
        "mean_sa": property_result["mean_sa"],
    }


def print_summary(results: dict[str, Any], novelty_computed: bool) -> None:
    print("[eval] molecular generation summary", flush=True)
    print(f"generated: {results['generated_path']}", flush=True)
    if results["train_path"] is not None:
        print(f"train: {results['train_path']}", flush=True)
    print(f"total_generated: {results['total_generated']}", flush=True)
    print(f"valid_count: {results['valid_count']}", flush=True)
    print(f"validity: {results['validity']:.4f}", flush=True)
    print(f"unique_valid_count: {results['unique_valid_count']}", flush=True)
    print(f"uniqueness: {results['uniqueness']:.4f}", flush=True)
    if novelty_computed:
        print(f"train_valid_unique_count: {results['train_valid_unique_count']}", flush=True)
        print(f"novel_unique_count: {results['novel_unique_count']}", flush=True)
        print(f"novelty: {results['novelty']:.4f}", flush=True)
    else:
        print("novelty: skipped (no train file)", flush=True)
    mean_qed = results["mean_qed"]
    mean_sa = results["mean_sa"]
    print(f"mean_qed: {mean_qed:.4f}" if mean_qed is not None else "mean_qed: null", flush=True)
    print(f"mean_sa: {mean_sa:.4f}" if mean_sa is not None else "mean_sa: null", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated SMILES strings with lightweight RDKit metrics")
    parser.add_argument("--generated", type=str, required=True, help="generated CSV or JSON file")
    parser.add_argument("--train", type=str, default=None, help="optional training/reference CSV or JSON file")
    parser.add_argument("--smiles-col", type=str, default="smiles", help="SMILES column/key for generated file")
    parser.add_argument(
        "--train-smiles-col",
        type=str,
        default=None,
        help="SMILES column/key for training file (defaults to --smiles-col)",
    )
    parser.add_argument("--output", type=str, required=True, help="output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_smiles = load_smiles_file(args.generated, smiles_col=args.smiles_col)
    train_smiles_col = args.train_smiles_col or args.smiles_col

    validity_result = compute_validity(generated_smiles)
    uniqueness_result = compute_uniqueness(validity_result["valid_canonical_smiles"])
    property_result = compute_property_scores(validity_result["valid_mols"])

    novelty_result: dict[str, Any] | None = None
    if args.train is not None:
        train_smiles = load_smiles_file(args.train, smiles_col=train_smiles_col)
        novelty_result = compute_novelty(uniqueness_result["unique_valid_smiles"], train_smiles)

    results = build_results_dict(
        generated_path=args.generated,
        train_path=args.train,
        smiles_col=args.smiles_col,
        train_smiles_col=train_smiles_col if args.train is not None else None,
        validity_result=validity_result,
        uniqueness_result=uniqueness_result,
        novelty_result=novelty_result,
        property_result=property_result,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(results, novelty_computed=(novelty_result is not None))


if __name__ == "__main__":
    main()
