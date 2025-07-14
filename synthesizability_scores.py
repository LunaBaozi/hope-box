import argparse
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))

from rdkit import Chem
from SA_Score import sascorer 
from NP_Score import npscorer
# from syba.syba import SybaClassifier

# Set the model path before importing the scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models', 'model.ckpt-10654.as_numpy.pickle')
os.environ['SCSCORER_MODEL_PATH'] = model_path

from scripts import scscorer_standalone 
from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder
from scripts.load_config_paths import PipelinePaths

# Initialize SCScorer
scscorer = scscorer_standalone.SCScorer()
scscorer.restore()

# Initialize SybaClassifier
# syba = SybaClassifier()
# syba.fitDefaultScore()

def sdf_to_mol(sdf_path, mol_path):
    """
    Converts an SDF file to a MOL file.
    Args:
        sdf_path (str): Path to the input SDF file.
        mol_path (str): Path to the output MOL file.
    """
    suppl = Chem.SDMolSupplier(sdf_path)
    for mol in suppl:
        if mol is not None:
            Chem.MolToMolFile(mol, mol_path)
            break  # Only write the first molecule

def calculate_sa_score(mol):
    """
    Calculates the Synthetic Accessibility Score (SA_Score) for a given molecule.
    Args:
        mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.
    Returns:
        float: The SA_Score score of the molecule.
    """
    return sascorer.calculateScore(mol) 

def calculate_sc_score(smi):
    """
    Calculates the Synthetic Complexity Score (SCScore) for a given molecule.
    Args:
        mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.
    Returns:
        float: The SCScore score of the molecule.
    """
    return scscorer.get_score_from_smi(smi)

def calculate_np_score(mol):
    """
    Calculates the Natural Product-likeness (NP_score) score for a given molecule.
    Args:
        mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.
    Returns:
        tuple: A tuple containing the NP_score and its confidence.
    """
    fscore = npscorer.readNPModel()
    score = npscorer.scoreMol(mol, fscore)
    confidence = npscorer.scoreMolWConfidence(mol, fscore)
    return score, confidence

# def calculate_syba_score(smi):
#     """
#     Calculates the SYnthetic Bayesian Accessibility Score (SYBA_score) for a given molecule.
#     Args:
#         mol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.
#     Returns:
#         float: The SYBA_score of the molecule.
#     """
#     return syba.predict(smi)

def calculate_scores(mols, smiles, filenames):
    results = []
    for mol, smi, fn in zip(mols, smiles, filenames):
        if mol is None:
            raise Exception('Not a valid mol')
        
        sa_score = calculate_sa_score(mol)
        np_score, _ = calculate_np_score(mol)
        (smi, sc_score) = calculate_sc_score(smi)
        # syba_score = calculate_syba_score(smi)
        
        results.append({
            'filename': fn,
            'smiles': smi,
            'len_smiles': len(smi),
            'SA_score': sa_score,
            'SCScore': sc_score,
            'NP_score': np_score,
            # 'Syba_score': syba_score
        })
    return pd.DataFrame(results)

if __name__ == '__main__':
    paths = PipelinePaths()

    parser = argparse.ArgumentParser(description='Wrapper for CADD pipeline targeting Aurora protein kinases.')
    parser.add_argument('--num_gen', type=int, required=False, default=0, help='Desired number of generated molecules (int, positive)')
    parser.add_argument('--epoch', type=int, required=False, default=0, help='Epoch number the model will use to generate molecules (int, 0-99)')
    parser.add_argument('--known_binding_site', type=str, required=False, default='0', help='Allow model to use binding site information (True, False)')
    parser.add_argument('--aurora', type=str, required=False, default='B', help='Aurora kinase type (str, A, B)')
    parser.add_argument('--pdbid', type=str, required=False, default='4af3', help='Aurora kinase type (str, A, B)')
    parser.add_argument('--experiment', type=str, required=False, default='default', help='Experiment name for output directory')
    parser.add_argument('--output_file', type=str, required=False, default=None, help='Output file path')
    args = parser.parse_args()

    epoch = args.epoch
    num_gen = args.num_gen
    known_binding_site = args.known_binding_site
    aurora = args.aurora
    pdbid = args.pdbid.lower() 
    experiment = args.experiment

    # output_csv = paths.synthesizability_output_path(epoch, num_gen, known_binding_site, pdbid, args.output_file)
    output_csv = paths.output_path(experiment, epoch, num_gen, known_binding_site, pdbid, args.output_file, 'synthesizability_scores')

    if epoch != 0:
        # Process generated molecules from GraphBP
        sdf_folder = paths.graphbp_sdf_path(epoch, num_gen, known_binding_site, pdbid)
        print(f"Loading molecules from: {sdf_folder}")
        
        mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)
        synth_scores = calculate_scores(mols, smiles, filenames)
        synth_scores.to_csv(output_csv, index=False)
        print(f"Processed {len(synth_scores)} generated molecules")

    else:
        # Process known Aurora kinase inhibitors
        aurora_data_file = paths.aurora_data_path(aurora)
        print(f"Loading Aurora kinase data from: {aurora_data_file}")
        
        mols, smiles, filenames, fps = read_aurora_kinase_interactions(aurora_data_file)
        synth_scores = calculate_scores(mols, smiles, filenames)
        synth_scores.to_csv(output_csv, index=False)
        print(f"Processed {len(synth_scores)} Aurora kinase inhibitors")

    print(f'Synthesizability scores saved to {output_csv}')


# NOTE: The SA_Score ranges from 1 to 10 with 1 being easy to make and 10 being hard to make.
# NOTE: The NP_Score ranges from -5 to 5 with -5 being easy to make and 5 being hard to make.
# NOTE: The SC_Score ranges from 1 to 5 with 1 being easy to make and 5 being hard to make.
# NOTE: While SYBA score can theoretically assume values between plus and minus infinity, 
#       a majority of compounds will have SYBA score between − 100 and +100 in real applications. 
#       It must be stressed here that the absolute value of the SYBA score is the measure of the 
#       confidence of the prediction and not of the degree of the synthetic accessibility.