from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DiffLinker generation without OpenBabel dependency")
    parser.add_argument("--difflinker-root", type=str, default="/home/yanglh/L/DiffLinker")
    parser.add_argument("--fragments", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--linker-size", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--anchors", type=str, default=None)
    return parser.parse_args()


def _ensure_repo_on_path(repo_root: Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _install_wandb_stub() -> None:
    if "wandb" in sys.modules:
        return
    module = types.ModuleType("wandb")
    module.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    module.log = lambda *args, **kwargs: None
    module.init = lambda *args, **kwargs: None
    module.finish = lambda *args, **kwargs: None
    module.Image = object
    sys.modules["wandb"] = module


def _install_lightweight_import_stubs() -> None:
    _install_wandb_stub()

    if "rdkit.six" not in sys.modules:
        rdkit_six_mod = types.ModuleType("rdkit.six")
        rdkit_six_mod.__spec__ = importlib.machinery.ModuleSpec("rdkit.six", loader=None)
        rdkit_six_mod.iteritems = lambda data: data.items()
        moves_mod = types.ModuleType("rdkit.six.moves")
        moves_mod.__spec__ = importlib.machinery.ModuleSpec("rdkit.six.moves", loader=None)
        moves_mod.cPickle = pickle
        rdkit_six_mod.moves = moves_mod
        sys.modules["rdkit.six"] = rdkit_six_mod
        sys.modules["rdkit.six.moves"] = moves_mod

    if "imageio" not in sys.modules and importlib.util.find_spec("imageio") is None:
        imageio_mod = types.ModuleType("imageio")
        imageio_mod.__spec__ = importlib.machinery.ModuleSpec("imageio", loader=None)
        imageio_mod.mimsave = lambda *args, **kwargs: None
        sys.modules["imageio"] = imageio_mod

    if "sklearn" not in sys.modules and importlib.util.find_spec("sklearn") is None:
        sklearn_mod = types.ModuleType("sklearn")
        sklearn_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        decomposition_mod = types.ModuleType("sklearn.decomposition")
        decomposition_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn.decomposition", loader=None)

        class _PCA:
            def __init__(self, *args, **kwargs):
                pass

            def fit_transform(self, x):
                return x

        decomposition_mod.PCA = _PCA
        sklearn_mod.decomposition = decomposition_mod
        sys.modules["sklearn"] = sklearn_mod
        sys.modules["sklearn.decomposition"] = decomposition_mod

    if "matplotlib" not in sys.modules and importlib.util.find_spec("matplotlib") is None:
        matplotlib_mod = types.ModuleType("matplotlib")
        matplotlib_mod.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None)
        pyplot_mod = types.ModuleType("matplotlib.pyplot")
        pyplot_mod.__spec__ = importlib.machinery.ModuleSpec("matplotlib.pyplot", loader=None)
        for name in [
            "figure",
            "subplot",
            "savefig",
            "close",
            "clf",
            "cla",
            "plot",
            "imshow",
            "axis",
            "gca",
        ]:
            setattr(pyplot_mod, name, lambda *args, **kwargs: None)
        matplotlib_mod.pyplot = pyplot_mod
        sys.modules["matplotlib"] = matplotlib_mod
        sys.modules["matplotlib.pyplot"] = pyplot_mod

    try:
        import pkg_resources
    except Exception:
        return

    real_get_distribution = pkg_resources.get_distribution

    class _DummyDistribution:
        version = "999.0.0"

    def _patched_get_distribution(dist):
        if str(dist) == "wandb":
            return _DummyDistribution()
        return real_get_distribution(dist)

    pkg_resources.get_distribution = _patched_get_distribution


def _read_molecule(path: str) -> Chem.Mol:
    if path.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol'):
        mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
    elif path.endswith('.mol2'):
        mol = Chem.MolFromMol2File(path, sanitize=False, removeHs=True)
    elif path.endswith('.sdf'):
        mol = Chem.SDMolSupplier(path, sanitize=False, removeHs=True)[0]
    else:
        raise ValueError('Unsupported fragment file extension')
    if mol is None:
        raise ValueError(f'Failed to read molecule from {path}')
    return Chem.RemoveAllHs(mol)


def _parse_linker_sampler(linker_size: str, device: torch.device, size_model_cls, const_mod):
    if linker_size.isdigit():
        linker_size_int = int(linker_size)

        def sample_fn(_data):
            return torch.ones(_data['positions'].shape[0], device=device, dtype=const_mod.TORCH_INT) * linker_size_int

        return sample_fn

    boundaries = [x.strip() for x in linker_size.split(',')]
    if len(boundaries) == 2 and boundaries[0].isdigit() and boundaries[1].isdigit():
        left = int(boundaries[0])
        right = int(boundaries[1])

        def sample_fn(_data):
            shape = (len(_data['positions']),)
            return torch.randint(left, right + 1, shape, device=device, dtype=const_mod.TORCH_INT)

        return sample_fn

    size_nn = size_model_cls.load_from_checkpoint(linker_size, map_location=device).eval().to(device)

    def sample_fn(_data):
        out, _ = size_nn.forward(_data, return_loss=False)
        probabilities = torch.softmax(out, dim=1)
        distribution = torch.distributions.Categorical(probs=probabilities)
        samples = distribution.sample()
        sizes = [size_nn.linker_id2size[label] for label in samples.detach().cpu().numpy()]
        return torch.tensor(sizes, device=samples.device, dtype=const_mod.TORCH_INT)

    return sample_fn


def main() -> None:
    args = parse_args()
    repo_root = Path(args.difflinker_root)
    _ensure_repo_on_path(repo_root)
    _install_lightweight_import_stubs()

    from src import const  # type: ignore
    from src.datasets import collate_with_fragment_edges, get_dataloader, parse_molecule  # type: ignore
    from src.lightning import DDPM  # type: ignore
    from src.linker_size_lightning import SizeClassifier  # type: ignore
    from src.molecule_builder import build_molecules  # type: ignore
    from src.utils import FoundNaNException  # type: ignore

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_fn = _parse_linker_sampler(args.linker_size, device, SizeClassifier, const)
    ddpm = DDPM.load_from_checkpoint(args.model, map_location=device).eval().to(device)
    if args.n_steps is not None:
        ddpm.edm.T = int(args.n_steps)

    if ddpm.center_of_mass == 'anchors' and args.anchors is None:
        raise ValueError('This checkpoint requires --anchors')

    mol = _read_molecule(args.fragments)
    positions, one_hot, charges = parse_molecule(mol, is_geom=ddpm.is_geom)
    fragment_mask = np.ones_like(charges)
    linker_mask = np.zeros_like(charges)
    anchor_flags = np.zeros_like(charges)
    if args.anchors is not None:
        for anchor in args.anchors.split(','):
            anchor_flags[int(anchor.strip()) - 1] = 1

    dataset = [{
        'uuid': '0',
        'name': '0',
        'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
        'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
        'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
        'anchors': torch.tensor(anchor_flags, dtype=const.TORCH_FLOAT, device=device),
        'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
        'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
        'num_atoms': len(positions),
    }] * int(args.n_samples)
    batch_size = min(int(args.n_samples), 64)
    dataloader = get_dataloader(dataset, batch_size=batch_size, collate_fn=collate_with_fragment_edges)

    fragment_stem = Path(args.fragments).stem
    sdf_paths: list[str] = []
    offset_idx = 0
    for data in dataloader:
        chain = None
        node_mask = None
        for _ in range(5):
            try:
                chain, node_mask = ddpm.sample_chain(data, sample_fn=sample_fn, keep_frames=1)
                break
            except FoundNaNException:
                continue
        if chain is None or node_mask is None:
            raise RuntimeError('Could not generate valid samples after 5 attempts')

        x = chain[0][:, :, :ddpm.n_dims]
        h = chain[0][:, :, ddpm.n_dims:]
        com_mask = data['fragment_mask'] if ddpm.center_of_mass == 'fragments' else data['anchors']
        pos_masked = data['positions'] * com_mask
        n = com_mask.sum(1, keepdims=True)
        mean = torch.sum(pos_masked, dim=1, keepdim=True) / n
        x = x + mean * node_mask

        mols = build_molecules(h, x, node_mask, is_geom=ddpm.is_geom)
        for local_idx, mol_out in enumerate(mols):
            out_path = output_dir / f'output_{offset_idx + local_idx}_{fragment_stem}_.sdf'
            writer = Chem.SDWriter(str(out_path))
            writer.write(mol_out)
            writer.close()
            sdf_paths.append(str(out_path))
        offset_idx += len(mols)

    print(f'[done] generated={len(sdf_paths)} output={output_dir}', flush=True)


if __name__ == '__main__':
    main()
