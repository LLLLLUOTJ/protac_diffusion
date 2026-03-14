from __future__ import annotations

import torch
from rdkit import Chem

from sampling.token_linker_codec import (
    build_token_templates_from_rows,
    canonical_smiles,
    decode_embedding_sequence_to_linker,
    decode_oriented_embedding_sequence_to_linker,
    mapped_tokens_from_token_sequence,
    mapped_tokens_from_oriented_sequence,
    normalize_mapped_token_sequence,
    stitch_mapped_tokens,
)


def test_stitch_mapped_tokens_roundtrip_linear() -> None:
    mapped_tokens = [
        "C([*:1])[*:3]",
        "C([*:3])[*:4]",
        "O([*:4])[*:5]",
        "C([*:5])[*:6]",
        "C([*:2])[*:6]",
    ]
    mol, reason = stitch_mapped_tokens(mapped_tokens)
    assert reason is None
    assert mol is not None

    expected = Chem.MolFromSmiles("[*:1]CCOCC[*:2]")
    assert expected is not None
    assert canonical_smiles(mol) == canonical_smiles(expected)


def test_templates_reconstruct_asymmetric_token_sequence() -> None:
    rows = [
        {
            "sample_id": "s1",
            "token_index": "0",
            "token_smiles": "*c1ccc(*)cc1",
            "token_smiles_with_maps": "c1cc([*:1])ccc1[*:3]",
        },
        {
            "sample_id": "s1",
            "token_index": "1",
            "token_smiles": "*O*",
            "token_smiles_with_maps": "O([*:3])[*:4]",
        },
        {
            "sample_id": "s1",
            "token_index": "2",
            "token_smiles": "*C*",
            "token_smiles_with_maps": "C([*:4])[*:5]",
        },
        {
            "sample_id": "s1",
            "token_index": "3",
            "token_smiles": "*C*",
            "token_smiles_with_maps": "C([*:2])[*:5]",
        },
    ]
    templates = build_token_templates_from_rows(rows)
    mapped_tokens = mapped_tokens_from_token_sequence(
        ["*c1ccc(*)cc1", "*O*", "*C*", "*C*"],
        templates,
        strict=True,
    )

    mol, reason = stitch_mapped_tokens(mapped_tokens)
    assert reason is None
    assert mol is not None

    expected = Chem.MolFromSmiles("[*:1]c1ccc(OCC[*:2])cc1")
    assert expected is not None
    assert canonical_smiles(mol) == canonical_smiles(expected)


def test_decode_embedding_sequence_to_linker() -> None:
    rows = [
        {
            "sample_id": "s1",
            "token_index": "0",
            "token_smiles": "*c1ccc(*)cc1",
            "token_smiles_with_maps": "c1cc([*:1])ccc1[*:3]",
        },
        {
            "sample_id": "s1",
            "token_index": "1",
            "token_smiles": "*O*",
            "token_smiles_with_maps": "O([*:3])[*:4]",
        },
        {
            "sample_id": "s1",
            "token_index": "2",
            "token_smiles": "*C*",
            "token_smiles_with_maps": "C([*:4])[*:5]",
        },
        {
            "sample_id": "s1",
            "token_index": "3",
            "token_smiles": "*C*",
            "token_smiles_with_maps": "C([*:2])[*:5]",
        },
    ]
    templates = build_token_templates_from_rows(rows)

    vocab_tokens = ["*C*", "*O*", "*c1ccc(*)cc1"]
    vocab_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    token_embeddings = torch.tensor(
        [
            [0.0, 0.1, 0.95],
            [0.05, 0.92, 0.02],
            [0.96, 0.02, 0.01],
            [0.93, 0.03, 0.02],
        ],
        dtype=torch.float32,
    )

    decoded = decode_embedding_sequence_to_linker(
        token_embeddings=token_embeddings,
        vocab_embeddings=vocab_embeddings,
        vocab_tokens=vocab_tokens,
        token_templates=templates,
    )

    assert decoded["reason"] is None
    assert decoded["token_smiles"] == ["*c1ccc(*)cc1", "*O*", "*C*", "*C*"]
    assert decoded["anchored_linker_smiles"] is not None

    expected = Chem.MolFromSmiles("[*:1]c1ccc(OCC[*:2])cc1")
    assert expected is not None
    assert str(decoded["anchored_linker_smiles"]) == canonical_smiles(expected)


def test_oriented_token_sequence_preserves_ambiguous_direction() -> None:
    mapped_tokens = [
        "N([*:1])[*:3]",
        "C([*:3])[*:4]",
        "C([*:4])[*:5]",
        "c1c([*:6])nnn1[*:5]",
        "C([*:6])[*:7]",
        "O([*:2])[*:7]",
    ]
    oriented_tokens = normalize_mapped_token_sequence(mapped_tokens)
    rebuilt_mapped = mapped_tokens_from_oriented_sequence(oriented_tokens)
    mol, reason = stitch_mapped_tokens(rebuilt_mapped)
    assert reason is None
    assert mol is not None

    expected = Chem.MolFromSmiles("c1c(CO[*:2])nnn1CCN[*:1]")
    assert expected is not None
    assert canonical_smiles(mol) == canonical_smiles(expected)


def test_decode_oriented_sequence_stops_at_pad_token() -> None:
    vocab_tokens = ["[*:1]C[*:2]", "[*:1]O[*:2]", "<PAD>"]
    vocab_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    token_embeddings = torch.tensor(
        [
            [0.99, 0.02, 0.0],
            [0.02, 0.98, 0.0],
            [0.0, 0.01, 0.99],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    decoded = decode_oriented_embedding_sequence_to_linker(
        token_embeddings=token_embeddings,
        vocab_embeddings=vocab_embeddings,
        vocab_tokens=vocab_tokens,
        stop_token="<PAD>",
    )

    assert decoded["reason"] is None
    assert decoded["stop_index"] == 2
    assert decoded["oriented_token_smiles"] == ["[*:1]C[*:2]", "[*:1]O[*:2]"]
    assert decoded["oriented_token_smiles_raw"] == ["[*:1]C[*:2]", "[*:1]O[*:2]", "<PAD>", "<PAD>"]
