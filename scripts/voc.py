import numpy as np

class VocabularyTokenizer():
    def __init__(self, vocs):
        """
        Parameters
        ----------
        vocs: array-like of str
        """
        vocs = sorted(list(vocs), key=lambda x:len(x), reverse=True)
        vocs_with_specials = ['<padding>', '<start>', '<end>'] + vocs
        self.voc_lens = np.sort(np.unique([len(voc) for voc in vocs]))[::-1]
        self.min_voc_len = self.voc_lens[-1]
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        self.voc2tok = {voc: tok for tok, voc in enumerate(vocs_with_specials)}
        print(self.voc2tok)
        self.tok2voc = np.array(vocs_with_specials)

    def tokenize(self, string):
        string_left = string
        toks = [self.start_token]
        while len(string_left) > 0:
            for voc_len in self.voc_lens:
                if string_left[:voc_len] in self.voc2tok:
                    toks.append(self.voc2tok[string_left[:voc_len]])
                    string_left = string_left[voc_len:]
                    break
                if voc_len == self.min_voc_len:
                    raise KeyError(f"Unknown keyward '{string_left}' in {string}")
        toks.append(self.end_token)
        return toks

    def detokenize(self, toks):
        """
        Parameters
        ----------
        toks: array_like of int

        Returns
        -------
        string: str
            detokenized string.
        """
        string = ""
        for tok in toks:
            if tok == self.end_token:
                break
            elif tok != self.start_token:
                string += self.tok2voc[tok]
        return string

    @property
    def voc_size(self):
        return len(self.tok2voc)

class OrderedTokenizer():

    def __init__(self, vocs):
        vocs[0] # check order exists in vocs
        vocs = ['<padding>', '<start>', '<end>'] + list(vocs)
        self.voc_lens = np.sort(np.unique([len(voc) for voc in vocs]))[::-1]
        self.min_voc_len = self.voc_lens[-1]
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        self.voc2tok = {voc: tok for tok, voc in enumerate(vocs)}
        self.tok2voc = np.array(vocs)

    def tokenize(self, string):
        string_left = string
        toks = [self.start_token]
        while len(string_left) > 0:
            for voc_len in self.voc_lens:
                if string_left[:voc_len] in self.voc2tok:
                    toks.append(self.voc2tok[string_left[:voc_len]])
                    string_left = string_left[voc_len:]
                    break
                if voc_len == self.min_voc_len:
                    raise KeyError(string)
        toks.append(self.end_token)
        return toks

    def detokenize(self, toks):
        """
        Parameters
        ----------
        toks: array_like of int

        Returns
        -------
        string: str
            detokenized string.
        """
        string = ""
        for tok in toks:
            if tok == self.end_token:
                break
            elif tok != self.start_token:
                string += self.tok2voc[tok]
        return string

    @property
    def voc_size(self):
        return len(self.tok2voc)
        
import pandas as pd
import pickle
from glob import glob
import os

for file_path in glob('../data/pubchem/*_pro_*.csv'):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    df2 = pd.read_csv(file_path)
    df2 = df2.drop_duplicates()

    df2 = df2[df2["canonical"].str.len() < 198]
    df2 = df2[df2["random"].str.len() < 198]

    #df_ff = df2[df2['canonical'].str.contains('TB')]
    #df2 = df2[~df2['canonical'].str.contains('TB')]

    #print(df_ff)
    e=df2["canonical"].tolist()

    tokens = []
    a=df2["random"].tolist()
    c=pd.read_csv("../data/large_vocs.csv")

    Token = VocabularyTokenizer(vocs=c["voc"])

    tokens = []
    for line in e:
        k = Token.tokenize(line)
        tokens.append(k)

    with open(f'../data/{file_name}_can.pkl', mode='wb') as fo:
        pickle.dump(tokens, fo)
    
    tokens = []
    for line in a:
        k = Token.tokenize(line)
        tokens.append(k)

    with open(f'../data/{file_name}_ran.pkl', mode='wb') as foo:
        pickle.dump(tokens, foo)
