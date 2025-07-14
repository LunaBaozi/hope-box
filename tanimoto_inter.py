import argparse
import pandas as pd
from rdkit.Chem import DataStructs
import numpy as np

from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder
from scripts.load_config_paths import PipelinePaths


def compute_tanimoto_scores(smi1, fn1, fps1, smi2, fn2, fps2):
    """
    Computes Tanimoto scores between two sets of fingerprints using BulkTanimotoSimilarity.
    Returns a DataFrame with columns: mol_1, smi_1, mol_2, smi_2, tanimoto.
    """
    results = []
    mat = np.zeros((len(fps1), len(fps2)))
    for i, fp1 in enumerate(fps1):
        scores = DataStructs.BulkTanimotoSimilarity(fp1, fps2)
        for j, tanimoto in enumerate(scores):
            mat[i, j] = scores[j]
            results.append({
                'filename': fn1[i],
                'smi_1': smi1[i],
                'mol_2': fn2[j],
                'smi_2': smi2[j],
                'tanimoto': tanimoto
            })
    return pd.DataFrame(results), mat

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

    num_gen = args.num_gen
    known_binding_site = args.known_binding_site
    epoch = args.epoch
    aurora = args.aurora
    pdbid = args.pdbid.lower()
    experiment = args.experiment

    output_csv = paths.output_path(experiment, epoch, num_gen, known_binding_site, pdbid, args.output_file, 'tanimoto_inter')

    other_aurora = 'B' if aurora == 'A' else 'A'
    aurora_data_file = paths.aurora_data_path(aurora)
    other_aurora_data_file = paths.aurora_data_path(other_aurora)

    if epoch != 0:
        # Process generated molecules from GraphBP
        sdf_folder = paths.graphbp_sdf_path(epoch, num_gen, known_binding_site, pdbid)
        print(f"Loading molecules from: {sdf_folder}")

        mols1, smiles1, filenames1, fps1 = load_mols_from_sdf_folder(sdf_folder)
        mols2, smiles2, filenames2, fps2 = read_aurora_kinase_interactions(aurora_data_file)        
        tanimoto, mat = compute_tanimoto_scores(smiles1, filenames1, fps1, smiles2, filenames2, fps2)
        tanimoto.to_csv(output_csv, index=False)

    else:
        # Calculating scores for Aurora inhibitors
        mols1, smiles1, filenames1, fps1 = read_aurora_kinase_interactions(aurora_data_file)
        mols2, smiles2, filenames2, fps2 = read_aurora_kinase_interactions(other_aurora_data_file)
        tanimoto, mat = compute_tanimoto_scores(smiles1, filenames1, fps1, smiles2, filenames2, fps2)
        tanimoto.to_csv(output_csv, index=False)

    print(f'Tanimoto inter results saved to {output_csv}')


    # # Mask for lower triangle
    # mask = np.tril(np.ones_like(mat, dtype=bool))
    # plt.figure(figsize=(8, 8))
    # plt.imshow(np.where(mask, mat, np.nan), cmap='viridis', vmin=0, vmax=1)
    # plt.colorbar(label='Tanimoto Similarity')
    # plt.title('Lower Triangular Tanimoto Similarity Heatmap')
    # plt.xlabel('Ligand Filename')
    # plt.ylabel('Ligand Filename')

    # # Only show labels for high-scoring pairs (tanimoto >= 0.5)
    # high_pairs = np.argwhere((mat >= 0.5) & mask & (np.arange(mat.shape[0])[:, None] > np.arange(mat.shape[1])))

    # n_labels1 = len(filenames1)
    # xtick_labels = ['' for _ in range(n_labels1)]
    # # ytick_labels = ['' for _ in range(n_labels1)]

    # n_labels2 = len(filenames2)
    # # xtick_labels = ['' for _ in range(n_labels1)]
    # ytick_labels = ['' for _ in range(n_labels2)]

    # # Set x-axis labels for position 1 of each high pair, y-axis for position 0
    # for i, j in high_pairs:
    #     ytick_labels[i] = filenames2[i]
    #     xtick_labels[j] = filenames1[j]

    # plt.xticks(
    #     ticks=range(n_labels1),
    #     labels=xtick_labels,
    #     rotation=45, ha='right', fontsize=7
    # )
    # plt.yticks(
    #     ticks=range(n_labels2),
    #     labels=ytick_labels,
    #     rotation=0, ha='right', fontsize=7
    # )
    # plt.tight_layout()
    # plt.savefig(os.path.join(image_dir, f'tanimoto_heatmap_inter_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.png'))
    # plt.close()
    # print(f'Lower triangular heatmap saved to {image_dir}/tanimoto_heatmap_inter_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.png')
