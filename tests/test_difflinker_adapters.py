from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from prepare_difflinker_inputs import prepare_inputs
from convert_difflinker_outputs import convert_rows
from sampling.linker_generation import assemble_full_molecule


def _build_fixture_row() -> dict[str, str]:
    left = 'CC(=O)N[*:1]'
    linker = '[*:1]CCO[*:2]'
    right = 'c1ccccc1[*:2]'
    full, reason = assemble_full_molecule(Chem.MolFromSmiles(left), Chem.MolFromSmiles(linker), Chem.MolFromSmiles(right))
    assert full is not None, reason
    return {
        'sample_id': '1',
        'protac_id': 'p1',
        'linker_id': 'l1',
        'full_protac_smiles': Chem.MolToSmiles(full, canonical=True),
        'anchored_linker_smiles': linker,
        'left_fragment_smiles': left,
        'right_fragment_smiles': right,
    }


def test_prepare_inputs_builds_manifest_and_fragment_sdf(tmp_path: Path) -> None:
    row = _build_fixture_row()
    weak_anchor = pd.DataFrame([row])
    selection = [{'sample_id': '1', 'source_dataset_index': 0}]
    prepared = prepare_inputs(weak_anchor, selection, out_dir=tmp_path, seed=13)
    assert len(prepared) == 1
    item = prepared[0]
    assert item.anchors.count(',') == 1
    assert item.linker_size == 3
    assert Path(item.fragments_path).exists()


def test_prepare_inputs_can_use_protac_sdf_by_id(tmp_path: Path) -> None:
    row = _build_fixture_row()
    weak_anchor = pd.DataFrame([row])
    selection = [{'sample_id': '1', 'source_dataset_index': 0}]

    full = Chem.MolFromSmiles(row['full_protac_smiles'])
    assert full is not None
    AllChem.Compute2DCoords(full)
    full.SetProp('_Name', row['protac_id'])
    full.SetProp('Smiles', row['full_protac_smiles'])
    protac_sdf = tmp_path / 'protac.sdf'
    writer = Chem.SDWriter(str(protac_sdf))
    writer.write(full)
    writer.close()

    prepared = prepare_inputs(weak_anchor, selection, out_dir=tmp_path, seed=13, protac_sdf=str(protac_sdf))
    assert len(prepared) == 1
    frag = Chem.SDMolSupplier(prepared[0].fragments_path, sanitize=False, removeHs=False)[0]
    assert frag is not None
    conf = frag.GetConformer()
    assert any(abs(conf.GetAtomPosition(i).z) > 1e-6 for i in range(frag.GetNumAtoms()))


def test_convert_rows_recovers_source_anchored_linker(tmp_path: Path) -> None:
    row = _build_fixture_row()
    weak_anchor = pd.DataFrame([row])
    selection = [{'sample_id': '1', 'source_dataset_index': 0}]
    prepared = prepare_inputs(weak_anchor, selection, out_dir=tmp_path, seed=13)
    manifest = [prepared[0].__dict__]

    sample_dir = Path(prepared[0].fragments_path).parent
    generated_dir = sample_dir / 'generated'
    generated_dir.mkdir(parents=True, exist_ok=True)
    full = Chem.MolFromSmiles(row['full_protac_smiles'])
    writer = Chem.SDWriter(str(generated_dir / 'output_0_fragments_.sdf'))
    writer.write(full)
    writer.close()

    rows = convert_rows(manifest, generated_subdir='generated')
    assert len(rows) == 1
    assert rows[0]['decode_reason'] is None
    assert Chem.MolToSmiles(Chem.MolFromSmiles(rows[0]['generated_anchored_linker_smiles']), canonical=True) == Chem.MolToSmiles(
        Chem.MolFromSmiles(row['anchored_linker_smiles']),
        canonical=True,
    )
