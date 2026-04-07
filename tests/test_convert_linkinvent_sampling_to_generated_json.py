from __future__ import annotations

from convert_linkinvent_sampling_to_generated_json import _label_fragment, _label_linker, _normalize_row


def test_label_fragment_adds_expected_anchor() -> None:
    assert _label_fragment("c1cc(*)ccc1", 1) == "c1cc([*:1])ccc1"
    assert _label_fragment("N[*:2]", 2) == "N[*:2]"


def test_label_linker_assigns_two_labels() -> None:
    assert _label_linker("*CC*") == "[*:1]CC[*:2]"
    assert _label_linker("[*]CC[*]") == "[*:1]CC[*:2]"


def test_normalize_row_builds_expected_fields() -> None:
    row = {
        "Warheads": "c1cc(*)ccc1|N*",
        "Linker": "*CC*",
        "SMILES": "c1cc(CCN)ccc1",
        "NLL": "5.0",
    }
    out = _normalize_row(row, row_index=3)
    assert out["sample_id"] == "linkinvent_0003"
    assert out["source_left_fragment_smiles"] == "c1cc([*:1])ccc1"
    assert out["source_right_fragment_smiles"] == "N[*:2]"
    assert out["generated_anchored_linker_smiles"] == "[*:1]CC[*:2]"
    assert out["generated_full_smiles"] == "c1cc(CCN)ccc1"
