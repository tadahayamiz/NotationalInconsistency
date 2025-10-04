import os
import numpy as np
import concurrent.futures as cf
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import Descriptors
from rdkit import Chem
from tqdm import tqdm


ORGANIC_ATOM_SET = set([1,5,6,7,8,9,15,16,17,35,53])
MIN_HEAVY_ATOM = 3
MAX_HEAVY_ATOM = 50
REMOVER = SaltRemover()
def curate_smiles(smiles_list):
    """
    1. Remove invalid SMILES.
    2. Remove chemicals with inorganic atom.
    3. Remove chemicals with too many/little heavy atoms
    4. Remove salt.
    5. Make random/canonical SMILES.

    Parameters
    ----------
    smiles_list: list of str.
        SMILES to be curated

    Returns
    -------
    randoms: list of str
        Randomized SMILES.
    canonicals: list of str
        Canonical SMILES.
    proper_mask: np.array[len(randoms)] of int
        Indices of extracted SMILES.
    n_valid:
        Number of valid SMILES in smiles_list.
    n_organic:
        Number of organic SMILES in smiles_list.
    """
    n_valid = n_organic = 0
    proper_mask = []
    canonicals = []
    randoms = []
    for i, smiles in enumerate(tqdm(smiles_list)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        n_valid += 1
        atom_set = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
        if not (atom_set <= ORGANIC_ATOM_SET):
            continue
        n_organic += 1
        n_heavy_atom = Descriptors.HeavyAtomCount(mol)
        if n_heavy_atom < MIN_HEAVY_ATOM or MAX_HEAVY_ATOM < n_heavy_atom:
            continue

        proper_mask.append(i)
        mol2 = REMOVER.StripMol(mol, dontRemoveEverything=True)
        smiles2 = Chem.MolToSmiles(mol2, isomericSmiles=True)
        if "." in smiles2:
            mol_frags = Chem.GetMolFrags(mol2, asMols=True)
            largest = None
            largest_size = 0
            for mol in mol_frags:
                size = mol.GetNumAtoms()
                if size > largest_size:
                    largest = mol
                    largest_size = size
            mol2 = largest
            smiles2 = Chem.MolToSmiles(largest)
        canonicals.append(smiles2)
        ans = list(range(mol2.GetNumAtoms()))
        np.random.shuffle(ans)
        random_mol = Chem.RenumberAtoms(mol2,ans)
        randoms.append(Chem.MolToSmiles(random_mol, canonical=False))
    return randoms, canonicals, np.array(proper_mask), n_valid, n_organic

# remove_saltの後にis_organic, heavy_atomを判断するようにした。
def curate_smiles2(smiles_list, check_is_organic=True, check_heavy_atom=True, show_tqdm=True):
    """
    1. Remove invalid SMILES.
    2. Remove salt.
    3. Remove chemicals with inorganic atom.
    4. Remove chemicals with too many/little heavy atoms
    5. Make random/canonical SMILES.

    Parameters
    ----------
    smiles_list: list of str.
        SMILES to be curated
    check_is_organic: bool
        If False, include inorganic molecules (= pass 3.)
    check_heavy_atom: bool
        If False, include molecules with too many/little heavy atoms (= pass 4.)

    Returns
    -------
    randoms: list of str
        Randomized SMILES.
    canonicals: list of str
        Canonical SMILES.
    proper_mask: np.array[len(randoms)] of int
        Indices of extracted SMILES.
    n_valid:
        Number of valid SMILES in smiles_list.
    n_organic:
        Number of organic SMILES in smiles_list.
    """
    n_valid = n_organic = 0
    proper_mask = []
    canonicals = []
    randoms = []
    for i, smiles in enumerate(tqdm(smiles_list) if show_tqdm else smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol2 = REMOVER.StripMol(mol, dontRemoveEverything=True)
        smiles2 = Chem.MolToSmiles(mol2, isomericSmiles=True)
        if "." in smiles2:
            mol_frags = Chem.GetMolFrags(mol2, asMols=True)
            largest = None
            largest_size = 0
            for mol in mol_frags:
                size = mol.GetNumAtoms()
                if size > largest_size:
                    largest = mol
                    largest_size = size
            mol2 = largest
            smiles2 = Chem.MolToSmiles(largest)

        n_valid += 1
        atom_set = {atom.GetAtomicNum() for atom in mol2.GetAtoms()}
        if not (atom_set <= ORGANIC_ATOM_SET) and check_is_organic:
            continue
        n_organic += 1
        n_heavy_atom = Descriptors.HeavyAtomCount(mol2)
        if n_heavy_atom < MIN_HEAVY_ATOM or MAX_HEAVY_ATOM < n_heavy_atom \
            and check_heavy_atom:
            continue

        proper_mask.append(i)
        canonicals.append(smiles2)
        ans = list(range(mol2.GetNumAtoms()))
        np.random.shuffle(ans)
        random_mol = Chem.RenumberAtoms(mol2,ans)
        randoms.append(Chem.MolToSmiles(random_mol, canonical=False))
    return randoms, canonicals, np.array(proper_mask), n_valid, n_organic

def curate_smiles2_mt(smiles_list, check_is_organic=True, check_heavy_atom=True, max_workers=1):
    chunk_size = (len(smiles_list) // max_workers)+1
    
    with cf.ProcessPoolExecutor(max_workers=max_workers) as e:
        futures = []
        for i_worker in range(max_workers):
            futures.append(e.submit(curate_smiles2, smiles_list[chunk_size*i_worker:chunk_size*(i_worker+1)],
                check_is_organic=check_is_organic, check_heavy_atom=check_heavy_atom, show_tqdm=False))
        randoms = []
        canonicals = []
        proper_mask = []
        n_valid = 0
        n_organic = 0
        for i_chunk, future in enumerate(futures):
            randoms_w, canonicals_w, proper_mask_w, n_valid_w, n_organic_w = \
                future.result()
            randoms += randoms_w
            canonicals += canonicals_w
            proper_mask.append(proper_mask_w+i_chunk*chunk_size)
            n_valid += n_valid_w
            n_organic += n_organic_w
    proper_mask = np.concatenate(proper_mask)
    return randoms, canonicals, proper_mask, n_valid, n_organic
            