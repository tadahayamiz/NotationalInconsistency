import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
import concurrent.futures as cf
from tqdm import tqdm
import time
import numpy as np


def process_data(chunk_id):
    data = pd.read_csv(f"./original_data/Pubchem_chunk_{chunk_id}.csv")
    df = pd.DataFrame(data)
    
    data2 = canonicalize_smiles(data["canonical"])
    data3 = randomize_smiles(data2)
    data = pd.concat([
        pd.DataFrame(data2, columns=["canonical"]),
        pd.DataFrame(data3, columns=["random"])
        ], axis=1)
    data.to_csv(f"./original_data/Pubchem_chunk_pro_{chunk_id}.csv")
    
def canonicalize_smiles(smiles):
    sm = [None] * len(smiles)
    for i, s in enumerate(tqdm(smiles)):
        if s is None:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        sm[i] = Chem.MolToSmiles(m, canonical=True)
    return sm

def randomize_smiles(smiles):
    sm = [None]*len(smiles)
    for i,s in enumerate(tqdm(smiles)):
        if s is None: continue
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        sm[i] = Chem.MolToSmiles(nm,canonical=False)
    return sm
    
def add_chirality(smiles, start_time):
    sm = [None]*len(smiles)
    for i,s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        rdmolops.AssignStereochemistryFrom3D(mol)
        sm[i] = Chem.MolToSmiles(mol)
        current_time = time.time()
        elapsed_time = current_time - start_time
    return sm
    
start_time = time.time()  
 
 
if __name__ == "__main__":
    start_time = time.time()

    with cf.ProcessPoolExecutor() as e:
        
        chunks = range(0, 5)
        
        e.map(process_data, chunks)
    
    print("Processing completed.")