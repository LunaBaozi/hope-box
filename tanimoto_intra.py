import os
import argparse
import itertools

# from rdkit import Chem
from rdkit.Chem import DataStructs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder


def compute_tanimoto_scores(smiles, filenames, fps):
    n = len(fps)
    mat = np.zeros((n, n))
    results = []

    for i, j in itertools.combinations(range(n), 2):
        score = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        mat[i, j] = score
        mat[j, i] = score 
        results.append({
            'mol_1': filenames[i],
            'smi_1': smiles[i],
            'mol_2': filenames[j],
            'smi_2': smiles[j],
            'tanimoto': score
        })
    return pd.DataFrame(results), mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wrapper for CADD pipeline targeting Aurora protein kinases.')
    parser.add_argument('--num_gen', type=int, required=False, default=0, help='Desired number of generated molecules (int, positive)')
    parser.add_argument('--epoch', type=int, required=False, default=0, help='Epoch number the model will use to generate molecules (int, 0-99)')
    parser.add_argument('--known_binding_site', type=str, required=False, default='0', help='Allow model to use binding site information (True, False)')
    parser.add_argument('--aurora', type=str, required=False, default='B', help='Aurora kinase type (str, A, B)')
    parser.add_argument('--pdbid', type=str, required=False, default='4af3', help='Aurora kinase type (str, A, B)')
    args = parser.parse_args()

    num_gen = args.num_gen
    known_binding_site = args.known_binding_site
    epoch = args.epoch
    aurora = args.aurora
    pdbid = args.pdbid.lower()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    sdf_folder = os.path.join(parent_dir, f'trained_model_reduced_dataset_100_epochs/gen_mols_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}/sdf')
    known_inhib_file = os.path.join(script_dir, f'data/aurora_kinase_{aurora}_interactions.csv')
    results_dir = os.path.join(script_dir, f'results_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}')
    image_dir = os.path.join(script_dir, f'results_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}/images')
    output_csv = os.path.join(results_dir, f'tanimoto_intra_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv')
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    if epoch != 0:
        # Calculating scores for generated molecules
        mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)
        tanimoto, mat = compute_tanimoto_scores(smiles, filenames, fps)
        tanimoto.to_csv(output_csv, index=False)

    else:
        # Calculating scores for Aurora inhibitors
        mols, smiles, filenames, fps = read_aurora_kinase_interactions(known_inhib_file)
        tanimoto, mat = compute_tanimoto_scores(smiles, filenames, fps)
        tanimoto.to_csv(output_csv, index=False)

    print(f'Tanimoto intra scores saved to {output_csv}')
    
    print('Generating lower triangular heatmap...')

    # Mask for lower triangle
    mask = np.tril(np.ones_like(mat, dtype=bool))
    plt.figure(figsize=(8, 8))
    plt.imshow(np.where(mask, mat, np.nan), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Tanimoto Similarity')
    plt.title('Lower Triangular Tanimoto Similarity Heatmap')
    plt.xlabel('Ligand Filename')
    plt.ylabel('Ligand Filename')

    # Only show labels for high-scoring pairs (tanimoto >= 0.5)
    high_pairs = np.argwhere((mat >= 0.5) & mask & (np.arange(mat.shape[0])[:, None] > np.arange(mat.shape[1])))

    n_labels = len(filenames)
    xtick_labels = ['' for _ in range(n_labels)]
    ytick_labels = ['' for _ in range(n_labels)]

    # Set x-axis labels for position 1 of each high pair, y-axis for position 0
    for i, j in high_pairs:
        ytick_labels[i] = filenames[i]
        xtick_labels[j] = filenames[j]

    plt.xticks(
        ticks=range(n_labels),
        labels=xtick_labels,
        rotation=45, ha='right', fontsize=7
    )
    plt.yticks(
        ticks=range(n_labels),
        labels=ytick_labels,
        rotation=0, ha='right', fontsize=7
    )
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f'tanimoto_heatmap_intra_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.png'))
    plt.close()
    print(f'Lower triangular heatmap saved to {image_dir}/tanimoto_heatmap_intra_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.png')
