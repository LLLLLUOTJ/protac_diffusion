from __future__ import annotations

import torch
from rdkit import Chem

from data.anchored_tensor_dataset import collate_graph_tensor_blocks, graph_block_from_mol
from data.weak_anchor_diffusion import dense_edge_tensor_from_graph
from sampling.linker_generation import (
    decode_generated_linker_batch,
    extract_anchored_component,
    fixed_edge_template_from_graph,
    update_linker_graph_from_dense_edges,
)


def test_extract_anchored_component_keeps_fragment_with_both_dummies() -> None:
    mol = Chem.MolFromSmiles("[*:1]CC[*:2].CC")
    assert mol is not None

    anchored, reason = extract_anchored_component(mol)
    assert reason is None
    assert anchored is not None
    smiles = Chem.MolToSmiles(anchored, canonical=True)
    assert "[*:1]" in smiles
    assert "[*:2]" in smiles
    assert "." not in smiles


def test_fixed_edge_template_only_keeps_dummy_incident_edges() -> None:
    mol = Chem.MolFromSmiles("[*:1]CCO[*:2]")
    assert mol is not None
    graph = graph_block_from_mol(mol, include_pair_mask=True)
    batch = collate_graph_tensor_blocks([graph])

    templ = fixed_edge_template_from_graph(batch, node_x=batch["x"])
    assert templ["edge_index"].shape[1] > 0

    dummy = templ["dummy_mask"].bool()
    src = templ["edge_index"][0]
    dst = templ["edge_index"][1]
    assert torch.all(dummy[src] | dummy[dst]).item()


def test_update_linker_graph_from_dense_edges_roundtrips_single_graph() -> None:
    mol = Chem.MolFromSmiles("[*:1]CCO[*:2]")
    assert mol is not None
    graph = graph_block_from_mol(mol, include_pair_mask=True)
    batch = collate_graph_tensor_blocks([graph])
    dense = dense_edge_tensor_from_graph(graph).unsqueeze(0)

    updated = update_linker_graph_from_dense_edges(
        linker_graph=batch,
        edge_tensor=dense,
        node_x=batch["x"],
        score_threshold=0.5,
    )
    assert updated["edge_index"].shape[1] >= 4

    decoded = decode_generated_linker_batch(
        node_x=batch["x"],
        edge_tensor=dense,
        linker_graph=batch,
        score_threshold=0.5,
    )
    assert len(decoded) == 1
    assert decoded[0]["mol"] is not None
    smiles = decoded[0]["anchored_linker_smiles"]
    assert smiles is not None
    assert "[*:1]" in smiles
    assert "[*:2]" in smiles
