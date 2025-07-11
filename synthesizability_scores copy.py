import argparse
import sys
import os
import pandas as pd
from pathlib import Path
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))

from rdkit import Chem
from SA_Score import sascorer 
from NP_Score import npscorer
# from syba.syba import SybaClassifier

# Set the model path before importing the scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# print(script_dir)
model_path = os.path.join(script_dir, 'models', 'model.ckpt-10654.as_numpy.pickle')
os.environ['SCSCORER_MODEL_PATH'] = model_path

from scripts import scscorer_standalone 
from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder

# Initialize SCScorer
scscorer = scscorer_standalone.SCScorer()
scscorer.restore()

# Initialize SybaClassifier
# syba = SybaClassifier()
# syba.fitDefaultScore()

def get_paths_from_config(config_path="../../../config/config.yaml"):
    """Load paths from pipeline configuration"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except:
        # Fallback to relative paths if config not found
        return None

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
    # Try to load pipeline config
    pipeline_config = get_paths_from_config()

    parser = argparse.ArgumentParser(description='Wrapper for CADD pipeline targeting Aurora protein kinases.')
    parser.add_argument('--num_gen', type=int, required=False, default=0, help='Desired number of generated molecules (int, positive)')
    parser.add_argument('--epoch', type=int, required=False, default=0, help='Epoch number the model will use to generate molecules (int, 0-99)')
    parser.add_argument('--known_binding_site', type=str, required=False, default='0', help='Allow model to use binding site information (True, False)')
    parser.add_argument('--aurora', type=str, required=False, default='B', help='Aurora kinase type (str, A, B)')
    parser.add_argument('--pdbid', type=str, required=False, default='4af3', help='Aurora kinase type (str, A, B)')
    parser.add_argument('--base_dir', type=str, required=False, default=None, 
                       help='Base directory of the pipeline')
    
    args = parser.parse_args()


    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    elif pipeline_config:
        base_dir = Path(pipeline_config['project_paths']['base_dir']).resolve()
    else:
        # Fallback: assume we're in external/hope-box and go up two levels
        base_dir = Path(__file__).parent.parent.parent
    
    # Build paths dynamically
    script_dir = Path(__file__).parent
    
    if pipeline_config and args.epoch != 0:
        # Use config-based paths
        graphbp_config = pipeline_config['modules']['graphbp']
        sdf_folder = (base_dir / 
                     graphbp_config['path'] / 
                     graphbp_config['trained_model_subdir'] / 
                     graphbp_config['output_pattern'].format(
                         epoch=args.epoch,
                         num_gen=args.num_gen,
                         known_binding_site=args.known_binding_site,
                         pdbid=args.pdbid
                     ))
    else:
        # Fallback paths
        sdf_folder = base_dir / f'external/graphbp/trained_model_reduced_dataset_100_epochs/gen_mols_epoch_{args.epoch}_mols_{args.num_gen}_bs_{args.known_binding_site}_pdbid_{args.pdbid}/sdf'
    

    # num_gen = args.num_gen
    # known_binding_site = args.known_binding_site
    # epoch = args.epoch
    # aurora = args.aurora
    # pdbid = args.pdbid.lower()  

    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    # sdf_folder = os.path.join(parent_dir, f'trained_model_reduced_dataset_100_epochs/gen_mols_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}/sdf')
    # known_inhib_file = os.path.join(script_dir, f'data/aurora_kinase_{aurora}_interactions.csv')
    # results_dir = os.path.join(script_dir, f'results_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}')
    # output_csv = os.path.join(results_dir, f'synthesizability_scores_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv')
    
    # os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    if epoch != 0:
        # Calculating scores for generated molecules
        mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)
        synth = calculate_scores(mols, smiles, filenames)
        synth.to_csv(output_csv, index=False)

    else:
        # Calculating scores for Aurora inhibitors
        mols, smiles, filenames, fps = read_aurora_kinase_interactions(known_inhib_file)
        synth = calculate_scores(mols, smiles, filenames)
        synth.to_csv(output_csv, index=False)

    print(f'Synthesizability scores saved to {output_csv}')


# NOTE: The SA_Score ranges from 1 to 10 with 1 being easy to make and 10 being hard to make.
# NOTE: The NP_Score ranges from -5 to 5 with -5 being easy to make and 5 being hard to make.
# NOTE: The SC_Score ranges from 1 to 5 with 1 being easy to make and 5 being hard to make.
# NOTE: While SYBA score can theoretically assume values between plus and minus infinity, 
#       a majority of compounds will have SYBA score between − 100 and +100 in real applications. 
#       It must be stressed here that the absolute value of the SYBA score is the measure of the 
#       confidence of the prediction and not of the degree of the synthetic accessibility.