from __future__ import annotations

from typing import List

from rdkit import Chem

from molgraph.featurize import decode_graph, encode_mol, find_anchor_indices


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"failed to parse: {smiles}"
    return Chem.MolToSmiles(mol, canonical=True)


def _canonical_smiles_from_mol(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def test_roundtrip_non_anchor_smiles() -> None:
    smiles_list: List[str] = [
        "CCO",
        "CC(C)O",
        "c1ccccc1",
        "C1CCCCC1",
    ]

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        graph = encode_mol(mol, include_original_smiles=True)
        mol2, reason = decode_graph(
            graph["x"],
            graph["edge_index"],
            graph["edge_attr"],
            node_type=graph["node_type"],
            meta=graph["meta"],
            return_reason=True,
        )
        assert mol2 is not None, reason
        assert _canonical_smiles_from_mol(mol2) == _canonical_smiles(smiles)


def test_anchor_roundtrip_preserves_dummy_map_numbers() -> None:
    smiles = "[*:1]CCOCC[*:2]"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None

    graph = encode_mol(mol, include_original_smiles=True)
    assert int((graph["node_type"] == 1).sum().item()) == 1
    assert int((graph["node_type"] == 2).sum().item()) == 1

    mol2, reason = decode_graph(
        graph["x"],
        graph["edge_index"],
        graph["edge_attr"],
        node_type=graph["node_type"],
        meta=graph["meta"],
        return_reason=True,
    )
    assert mol2 is not None, reason

    dummy_maps = sorted([atom.GetAtomMapNum() for atom in mol2.GetAtoms() if atom.GetAtomicNum() == 0])
    assert dummy_maps == [1, 2]


def test_find_anchor_indices_fallback_to_dummy_atoms() -> None:
    mol = Chem.MolFromSmiles("*CCOCC*")
    assert mol is not None
    left, right = find_anchor_indices(mol)
    assert left is not None and right is not None
    assert left != right

    graph = encode_mol(mol)
    assert int((graph["node_type"] == 1).sum().item()) == 1
    assert int((graph["node_type"] == 2).sum().item()) == 1


def test_encode_with_explicit_anchor_indices() -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    graph = encode_mol(mol, explicit_anchor_indices=(0, 2))
    assert graph["node_type"].tolist() == [1, 0, 2]
