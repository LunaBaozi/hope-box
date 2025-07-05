#!/usr/bin/env python
"""
Lipinski's Rule of Five calculation from:
https://gist.github.com/strets123/fdc4db6d450b66345f46
"""

import os
import argparse
import pandas as pd

# from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors

from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder


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
    parser = argparse.ArgumentParser(description='Wrapper for CADD pipeline targeting Aurora protein kinases.')
    parser.add_argument('--num_gen', type=int, required=False, default=0, help='Desired number of generated molecules (int, positive)')
    parser.add_argument('--epoch', type=int, required=False, default=0, help='Epoch number the model will use to generate molecules (int, 0-99)')
    parser.add_argument('--known_binding_site', type=str, required=False, default='0', help='Allow model to use binding site information (True, False)')
    parser.add_argument('--aurora', type=str, required=False, default='B', help='Aurora kinase type (str, A, B)')
    parser.add_argument('--pdbid', type=str, required=False, default='4af3', help='Aurora kinase type (str, A, B)')
    args = parser.parse_args()

    epoch = args.epoch
    num_gen = args.num_gen
    known_binding_site = args.known_binding_site
    aurora = args.aurora
    pdbid = args.pdbid.lower()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    sdf_folder = os.path.join(parent_dir, f'trained_model_reduced_dataset_100_epochs/gen_mols_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}/sdf')
    known_inhib_file = os.path.join(script_dir, f'data/aurora_kinase_{aurora}_interactions.csv')
    results_dir = os.path.join(script_dir, f'results_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}')
    output_csv = os.path.join(results_dir, f'lipinski_pass_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv')

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if epoch != 0:
        # Calculating scores for generated molecules
        mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)
        lipinski = evaluate_lipinski_rules(mols, smiles, filenames)
        lipinski.to_csv(output_csv, index=False)

    else:
        # Calculating scores for Aurora inhibitors
        mols, smiles, filenames, fps = read_aurora_kinase_interactions(known_inhib_file)
        lipinski = evaluate_lipinski_rules(mols, smiles, filenames)
        lipinski.to_csv(output_csv, index=False)

    print(f'Lipinski\'s Ro5 results saved to {output_csv}')