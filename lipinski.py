#!/usr/bin/env python
"""
Lipinski's Rule of Five calculation from:
https://gist.github.com/strets123/fdc4db6d450b66345f46
"""
import argparse
import pandas as pd

# from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors

from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder
from scripts.load_config_paths import PipelinePaths

def lipinski_trial(mol): 
    passed = []
    failed = []
    num_hdonors = Lipinski.NumHDonors(mol)
    num_hacceptors = Lipinski.NumHAcceptors(mol)
    mol_weight = Descriptors.MolWt(mol)
    mol_logp = Crippen.MolLogP(mol)
    
    if num_hdonors > 5:
        failed.append('Over 5 H-bond donors, found %s' % num_hdonors)
    else:
        passed.append('Found %s H-bond donors' % num_hdonors)
    if num_hacceptors > 10:
        failed.append('Over 10 H-bond acceptors, found %s' % num_hacceptors)
    else:
        passed.append('Found %s H-bond acceptors' % num_hacceptors)
    if mol_weight >= 500:
        failed.append('Molecular weight over 500, calculated %s' % mol_weight)
    else:
        passed.append('Molecular weight: %s' % mol_weight)
    if mol_logp >= 5:
        failed.append('Log partition coefficient over 5, calculated %s' % mol_logp)
    else:
        passed.append('Log partition coefficient: %s' % mol_logp)
    return passed, failed 

def evaluate_lipinski_rules(mols, smiles, filenames):
    results = []
    for mol, smi, fn in zip(mols, smiles, filenames):
        if mol is None:
            raise Exception('Not a valid mol')
        try:
            passed, failed = lipinski_trial(mol)
            results.append({
                'filename': fn,
                'smiles': smi,
                'passed': '; '.join(passed),
                'failed': '; '.join(failed)
            })
        except Exception as e:
            results.append({
                'filename': fn,
                'smiles': smi,
                'passed': '',
                'failed': f'Error: {str(e)}'
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
    parser.add_argument('--experiment', type=str, required=False, default='default', help='Aurora kinase type (str, A, B)')
    parser.add_argument('--output_file', type=str, required=False, default=None, help='Output file path')
    args = parser.parse_args()

    epoch = args.epoch
    num_gen = args.num_gen
    known_binding_site = args.known_binding_site
    aurora = args.aurora
    pdbid = args.pdbid.lower()
    experiment = args.experiment

    # output_csv = paths.output_path(epoch, num_gen, known_binding_site, pdbid, args.output_file) 
    output_csv = paths.output_path(experiment, epoch, num_gen, known_binding_site, pdbid, args.output_file, 'lipinski_pass')

    if epoch != 0:
        # Process generated molecules from GraphBP
        sdf_folder = paths.graphbp_sdf_path(epoch, num_gen, known_binding_site, pdbid)
        print(f"Loading molecules from: {sdf_folder}")

        mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)
        lipinski = evaluate_lipinski_rules(mols, smiles, filenames)
        lipinski.to_csv(output_csv, index=False)

    else:
        # Process known Aurora kinase inhibitors
        aurora_data_file = paths.aurora_data_path(aurora)
        print(f"Loading Aurora kinase data from: {aurora_data_file}")
        
        mols, smiles, filenames, fps = read_aurora_kinase_interactions(aurora_data_file)
        lipinski = evaluate_lipinski_rules(mols, smiles, filenames)
        lipinski.to_csv(output_csv, index=False)

    print(f'Lipinski\'s Ro5 results saved to {output_csv}')