import argparse

from rdkit.Chem import DataStructs
import numpy as np
import pandas as pd

from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder
from scripts.load_config_paths import PipelinePaths

def compute_tanimoto_scores(smiles, filenames, fps):
    """Compute intra-molecular Tanimoto similarity scores"""

    n = len(fps)
    # tanimoto_matrix = []
    mat = np.zeros((n, n))
    results = []
    
    # Compute pairwise Tanimoto similarities
    for i in range(n):
        # row = []
        for j in range(i+1, n):
            if fps[i] is not None and fps[j] is not None:
                score = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                mat[i, j] = score
                mat[j, i] = score
                # row.append(score)
            else:
                mat[i, j] = 0.0
                mat[j, i] = 0.0
                # row.append(0.0)  # Default for invalid molecules
            results.append({
                'mol_1': filenames[i],
                'smi_1': smiles[i],
                'mol_2': filenames[j],
                'smi_2': smiles[j],
                'tanimoto': score
            })
        # tanimoto_matrix.append(row)
        np.fill_diagonal(mat, 1.0)  # Self-similarity is always 1.0
    
    # Calculate average similarity for each molecule (excluding self-similarity)
    # avg_similarities = []
    # for i in range(n):
    #     similarities = [tanimoto_matrix[i][j] for j in range(n) if i != j]
    #     avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    #     avg_similarities.append(avg_sim)
    
    return pd.DataFrame(results), mat  #avg_similarities, tanimoto_matrix

if __name__ == '__main__':
    paths = PipelinePaths()

    parser = argparse.ArgumentParser(description='Wrapper for CADD pipeline targeting Aurora protein kinases.')
    parser.add_argument('--num_gen', type=int, required=False, default=0, help='Desired number of generated molecules (int, positive)')
    parser.add_argument('--epoch', type=int, required=False, default=0, help='Epoch number the model will use to generate molecules (int, 0-99)')
    parser.add_argument('--known_binding_site', type=str, required=False, default='0', help='Allow model to use binding site information (True, False)')
    parser.add_argument('--aurora', type=str, required=False, default='B', help='Aurora kinase type (str, A, B)')
    parser.add_argument('--pdbid', type=str, required=False, default='4af3', help='Aurora kinase type (str, A, B)')
    parser.add_argument('--output_file', type=str, required=False, default=None, help='Output file path')    
    args = parser.parse_args()

    num_gen = args.num_gen
    known_binding_site = args.known_binding_site
    epoch = args.epoch
    aurora = args.aurora
    pdbid = args.pdbid.lower()

    output_csv = paths.output_path(epoch, num_gen, known_binding_site, pdbid, args.output_file, 'tanimoto_intra')

    if epoch != 0:
        # Process generated molecules from GraphBP
        sdf_folder = paths.graphbp_sdf_path(epoch, num_gen, known_binding_site, pdbid)
        print(f"Loading molecules from: {sdf_folder}")

        mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)
        tanimoto, mat = compute_tanimoto_scores(smiles, filenames, fps)
        tanimoto.to_csv(output_csv, index=False)

    else:
        # Process known Aurora kinase inhibitors
        aurora_data_file = paths.aurora_data_path(aurora)
        print(f"Loading Aurora kinase data from: {aurora_data_file}")
        
        mols, smiles, filenames, fps = read_aurora_kinase_interactions(aurora_data_file)
        tanimoto, mat = compute_tanimoto_scores(smiles, filenames, fps)
        tanimoto.to_csv(output_csv, index=False)

    print(f'Tanimoto intra results saved to {output_csv}')
    
    # print('Generating lower triangular heatmap...')

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

    # n_labels = len(filenames)
    # xtick_labels = ['' for _ in range(n_labels)]
    # ytick_labels = ['' for _ in range(n_labels)]

    # # Set x-axis labels for position 1 of each high pair, y-axis for position 0
    # for i, j in high_pairs:
    #     ytick_labels[i] = filenames[i]
    #     xtick_labels[j] = filenames[j]

    # plt.xticks(
    #     ticks=range(n_labels),
    #     labels=xtick_labels,
    #     rotation=45, ha='right', fontsize=7
    # )
    # plt.yticks(
    #     ticks=range(n_labels),
    #     labels=ytick_labels,
    #     rotation=0, ha='right', fontsize=7
    # )
    # plt.tight_layout()
    # plt.savefig(os.path.join(image_dir, f'tanimoto_heatmap_intra_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.png'))
    # plt.close()
    # print(f'Lower triangular heatmap saved to {image_dir}/tanimoto_heatmap_intra_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.png')
