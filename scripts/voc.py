# scripts/voc.py
import argparse
from pathlib import Path
from glob import glob
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


class VocabularyTokenizer:
    """
    Greedy longest-match tokenizer over a given vocabulary.
    Three special tokens are prepended: <padding>=0, <start>=1, <end>=2.
    """

    def __init__(self, vocs):
        """
        Parameters
        ----------
        vocs : iterable of str
            Base vocabulary items (without special tokens).
        """
        vocs = sorted(list(vocs), key=lambda x: len(x), reverse=True)
        vocs_with_specials = ["<padding>", "<start>", "<end>"] + vocs
        self.voc_lens = np.sort(np.unique([len(v) for v in vocs]))[::-1]
        self.min_voc_len = self.voc_lens[-1]
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        self.voc2tok = {v: i for i, v in enumerate(vocs_with_specials)}
        self.tok2voc = np.array(vocs_with_specials)

    def tokenize(self, string: str):
        """Convert a string into a list of token IDs using greedy longest match."""
        string_left = string
        toks = [self.start_token]
        while len(string_left) > 0:
            for L in self.voc_lens:
                piece = string_left[:L]
                if piece in self.voc2tok:
                    toks.append(self.voc2tok[piece])
                    string_left = string_left[L:]
                    break
                if L == self.min_voc_len:
                    # Raise with context for easier debugging
                    raise KeyError(f"Unknown fragment '{string_left}' in: {string}")
        toks.append(self.end_token)
        return toks

    def detokenize(self, toks):
        """Inverse of tokenize (stops at <end>)."""
        out = []
        for t in toks:
            if t == self.end_token:
                break
            if t != self.start_token:
                out.append(self.tok2voc[t])
        return "".join(out)

    @property
    def voc_size(self):
        return len(self.tok2voc)


class OrderedTokenizer:
    """
    Tokenizer that respects the given order of vocabulary (first element must exist).
    Three special tokens are prepended: <padding>=0, <start>=1, <end>=2.
    """

    def __init__(self, vocs):
        vocs[0]  # ensure the iterable is indexable/non-empty
        vocs = ["<padding>", "<start>", "<end>"] + list(vocs)
        self.voc_lens = np.sort(np.unique([len(v) for v in vocs]))[::-1]
        self.min_voc_len = self.voc_lens[-1]
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        self.voc2tok = {v: i for i, v in enumerate(vocs)}
        self.tok2voc = np.array(vocs)

    def tokenize(self, string):
        string_left = string
        toks = [self.start_token]
        while len(string_left) > 0:
            for L in self.voc_lens:
                piece = string_left[:L]
                if piece in self.voc2tok:
                    toks.append(self.voc2tok[piece])
                    string_left = string_left[L:]
                    break
                if L == self.min_voc_len:
                    raise KeyError(string)
        toks.append(self.end_token)
        return toks

    def detokenize(self, toks):
        out = []
        for t in toks:
            if t == self.end_token:
                break
            if t != self.start_token:
                out.append(self.tok2voc[t])
        return "".join(out)

    @property
    def voc_size(self):
        return len(self.tok2voc)


def _resolve_vocab_path(input_dir: Path, vocab_path: Path | None) -> Path:
    """
    Resolve the path to large_vocs.csv.
    Priority:
      1) --vocab if provided
      2) input_dir / 'large_vocs.csv'
      3) input_dir.parent / 'large_vocs.csv'
    """
    if vocab_path is not None:
        return vocab_path
    cand1 = input_dir / "large_vocs.csv"
    if cand1.exists():
        return cand1
    cand2 = input_dir.parent / "large_vocs.csv"
    if cand2.exists():
        return cand2
    raise FileNotFoundError(
        f"Could not find 'large_vocs.csv'. Tried:\n - {cand1}\n - {cand2}\n"
        "Specify it explicitly via --vocab /path/to/large_vocs.csv"
    )


def _load_vocab(vocab_csv: Path):
    df_v = pd.read_csv(vocab_csv)
    if "voc" not in df_v.columns:
        raise ValueError(f"'voc' column not found in {vocab_csv}")
    return df_v["voc"]


def _iter_input_csvs(input_dir: Path, pattern: str):
    """
    Iterate input CSV files under input_dir matching the pattern.
    Default pattern is '*_pro_*.csv' to align with pubchem.py outputs.
    """
    for p in sorted(input_dir.glob(pattern)):
        if p.is_file() and p.suffix.lower() == ".csv":
            yield p


def _tokenize_dataframe(df: pd.DataFrame, tokenizer: VocabularyTokenizer,
                        length_max: int = 198) -> tuple[list[list[int]], list[list[int]]]:
    """
    Tokenize 'canonical' and 'random' columns from a DataFrame after filtering.
    """
    if "canonical" not in df.columns or "random" not in df.columns:
        raise ValueError("Both 'canonical' and 'random' columns are required in input CSVs.")

    # Drop duplicates then filter by length
    df = df.drop_duplicates()
    df = df[df["canonical"].str.len() < length_max]
    df = df[df["random"].str.len() < length_max]

    can_list = []
    ran_list = []
    for s in tqdm(df["canonical"].tolist(), desc="tokenize canonical", leave=False):
        can_list.append(tokenizer.tokenize(s))
    for s in tqdm(df["random"].tolist(), desc="tokenize random", leave=False):
        ran_list.append(tokenizer.tokenize(s))
    return can_list, ran_list


def _save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize canonical/random SMILES CSVs produced by pubchem.py."
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Directory containing CSV files (default pattern '*_pro_*.csv')."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Directory to write pickles (default: input_dir)."
    )
    parser.add_argument(
        "--vocab", type=Path, default=None,
        help="Path to large_vocs.csv (default: input_dir/large_vocs.csv "
             "or input_dir.parent/large_vocs.csv)."
    )
    parser.add_argument(
        "--pattern", type=str, default="*_pro_*.csv",
        help="Glob pattern for input CSVs (default: '*_pro_*.csv')."
    )
    parser.add_argument(
        "--length_max", type=int, default=198,
        help="Max string length (exclusive) to keep (default: 198)."
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or input_dir
    vocab_csv = _resolve_vocab_path(input_dir, args.vocab)

    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Vocab CSV : {vocab_csv}")
    print(f"Pattern   : {args.pattern}")
    print(f"length_max: {args.length_max}")

    # Load vocabulary and initialize tokenizer
    voc_series = _load_vocab(vocab_csv)
    tokenizer = VocabularyTokenizer(voc_series)

    # Process each CSV
    for csv_path in _iter_input_csvs(input_dir, args.pattern):
        # Example: Pubchem_chunk_pro_0.csv -> base 'Pubchem_chunk_pro_0'
        base = csv_path.stem
        print(f"[file] {csv_path.name}")

        df = pd.read_csv(csv_path)
        can_tokens, ran_tokens = _tokenize_dataframe(df, tokenizer, length_max=args.length_max)

        # Save pickles alongside output_dir with suffixes
        can_pkl = output_dir / f"{base}_can.pkl"
        ran_pkl = output_dir / f"{base}_ran.pkl"
        _save_pickle(can_tokens, can_pkl)
        _save_pickle(ran_tokens, ran_pkl)
        print(f"  -> wrote: {can_pkl.name}, {ran_pkl.name}")


if __name__ == "__main__":
    main()
