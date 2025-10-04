# scripts/pubchem.py
import argparse
import re
from pathlib import Path
import concurrent.futures as cf
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops


def canonicalize_smiles(smiles):
    """
    Canonicalize a list of SMILES strings using RDKit.
    Invalid or None entries are skipped.
    """
    sm = [None] * len(smiles)
    for i, s in enumerate(tqdm(smiles, desc="canonicalize", leave=False)):
        if not s:
            continue
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        sm[i] = Chem.MolToSmiles(m, canonical=True)
    return sm


def randomize_smiles(smiles, seed=None):
    """
    Generate randomized SMILES by shuffling atom orders.
    This preserves chemical identity but produces non-canonical strings.
    """
    rng = np.random.default_rng(seed)
    sm = [None] * len(smiles)
    for i, s in enumerate(tqdm(smiles, desc="randomize", leave=False)):
        if not s:
            continue
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        order = np.arange(m.GetNumAtoms())
        rng.shuffle(order)
        nm = Chem.RenumberAtoms(m, order.tolist())
        sm[i] = Chem.MolToSmiles(nm, canonical=False)
    return sm


def add_chirality(smiles, start_time=None):
    """
    Attempt to add chirality information by embedding molecules in 3D,
    adding hydrogens, and assigning stereochemistry.
    Not used in the main pipeline but available for further extensions.
    """
    sm = [None] * len(smiles)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        rdmolops.AssignStereochemistryFrom3D(mol)
        sm[i] = Chem.MolToSmiles(mol)
        if start_time:
            _ = time.time() - start_time
    return sm


def process_one_csv(in_path: Path, out_path: Path, seed=None):
    """
    Process one CSV file:
      - Read input (expects 'canonical' column).
      - Canonicalize SMILES.
      - Generate randomized SMILES.
      - Write canonical and randomized columns to output CSV.
    """
    df = pd.read_csv(in_path)
    if "canonical" not in df.columns:
        raise ValueError(f'"canonical" column not found in {in_path}')
    can = canonicalize_smiles(df["canonical"])
    ran = randomize_smiles(can, seed=seed)
    out_df = pd.DataFrame({"canonical": can, "random": ran})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)


def discover_chunks(input_dir: Path, start=None, end=None):
    """
    Discover all Pubchem_chunk_*.csv files in input_dir.
    Return a list of tuples (chunk_id, path).
    Optionally restrict the range using start and end (inclusive).
    """
    pat = re.compile(r"Pubchem_chunk_(\d+)\.csv$")
    items = []
    for p in input_dir.glob("Pubchem_chunk_*.csv"):
        m = pat.search(p.name)
        if not m:
            continue
        cid = int(m.group(1))
        if (start is not None and cid < start) or (end is not None and cid > end):
            continue
        items.append((cid, p))
    return sorted(items, key=lambda x: x[0])


def main():
    """
    Main function:
      - Parse arguments.
      - Discover available chunk files.
      - Process each chunk in parallel or sequentially.
    """
    ap = argparse.ArgumentParser(description="Canonicalize & randomize SMILES for PubChem chunks.")
    ap.add_argument("--input_dir", required=True, type=Path, help="Directory containing Pubchem_chunk_*.csv files")
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="Directory for processed CSVs (default: input_dir/result)")
    ap.add_argument("--start", type=int, default=None, help="Minimum chunk ID to process (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="Maximum chunk ID to process (inclusive)")
    ap.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 or 1 = sequential)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for SMILES randomization (per chunk)")
    args = ap.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or (input_dir / "result")

    pairs = discover_chunks(input_dir, start=args.start, end=args.end)
    if not pairs:
        raise SystemExit(f"No matching CSVs found in {input_dir}/Pubchem_chunk_*.csv")

    print(f"Found {len(pairs)} chunks in {input_dir}")
    print(f"Output directory: {output_dir}")
    start_time = time.time()

    def _task(cid_path):
        cid, in_path = cid_path
        out_path = output_dir / f"Pubchem_chunk_pro_{cid}.csv"
        print(f"[chunk {cid}] {in_path.name} -> {out_path.name}")
        process_one_csv(in_path, out_path, seed=None if args.seed is None else args.seed + cid)

    if args.workers and args.workers > 1:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            list(tqdm(ex.map(_task, pairs), total=len(pairs), desc="chunks"))
    else:
        for item in tqdm(pairs, total=len(pairs), desc="chunks"):
            _task(item)

    print(f"Processing completed in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
