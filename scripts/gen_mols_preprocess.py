import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

def load_mols_from_sdf_folder(sdf_folder):
    mols = []
    smiles = []
    filenames = []

    folder_path = Path(sdf_folder)

    if not folder_path.exists():
        print(f"Warning: SDF folder not found: {folder_path}")
        return mols, smiles, filenames
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.sdf'):
            mol = Chem.SDMolSupplier(os.path.join(folder_path, filename))[0]
            if mol is not None:
                mols.append(mol)
                smiles.append(Chem.MolToSmiles(mol))
                filenames.append(filename)
            else:
                print(f"Warning: Could not parse molecule from {filename}")

    # Generate Morgan fingerprints
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(mol) for mol in mols]

    print(f"Loaded {len(mols)} molecules from {folder_path}")
    return mols, smiles, filenames, fps