import csv
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def read_aurora_kinase_interactions(csv_path, smiles_col='smiles'):

    data = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', skipinitialspace=True)
        for row in reader:
            cleaned_row = {k: v.strip() if isinstance(v, str) else v for k, v in row.items()}
            data.append(cleaned_row)
    df = pd.DataFrame(data)
    smiles = df['smiles']
    mols = [Chem.MolFromSmiles(sm) for sm in df[smiles_col] if Chem.MolFromSmiles(sm) is not None]
    filenames = df['ligand']
    
    # Generate Morgan fingerprints
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(mol) for mol in mols]

    return mols, smiles, filenames, fps
