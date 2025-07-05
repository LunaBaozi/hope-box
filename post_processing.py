import os, argparse
import pandas as pd

from rdkit import Chem

from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder



def merge_on_smiles(synth_path, 
                    lipinski_path,
                    tanimoto_path,
                    output_path):
    """
    Merges three CSV files on the 'smiles' column, sorts by 'tanimoto' in decreasing order, and saves the result.
    """

    scores_df = pd.read_csv(synth_path)
    lipinski_df = pd.read_csv(lipinski_path)
    tanimoto_df = pd.read_csv(tanimoto_path)
    merged_df = pd.merge(scores_df, tanimoto_df, on='filename', how='outer')
    merged_df = pd.merge(merged_df, lipinski_df, on='filename', how='left')
    merged_df = merged_df.sort_values(by='tanimoto', ascending=False)

    print(set(scores_df['filename']) == set(tanimoto_df['filename']))

    desired_columns = [
        'filename', 'smiles', 'len_smiles', 'SA_score', 'NP_score', 'SCScore',
        'mol_2', 'tanimoto', 'passed', 'failed'
    ] 
    columns_to_use = [col for col in desired_columns if col in merged_df.columns]
    merged_df = merged_df[columns_to_use]

    merged_df.to_csv(output_path, index=False)
    return merged_df


def export_top_100_tanimoto(df):
    return df.sort_values(by='tanimoto', ascending=False).head(100)


def export_top_50_sa_score(input_df):
    df_sorted = input_df.sort_values(by='SA_score', ascending=True)    
    return df_sorted.drop_duplicates(subset='filename').head(50)



def copy_top_50_ligands(mols, top_50_sa_score, dest_ligand_dir):
    """
    Copies ligand files for the top 50 molecules (by filename) from SDF source,
    or generates SDFs from SMILES if a CSV is provided.
    """
    os.makedirs(dest_ligand_dir, exist_ok=True)

    mols_map = dict(zip(top_50_sa_score['filename'], mols))
    for fname, mol in mols_map.items():
        if mol is not None:
            mol = Chem.AddHs(mol)
            # Ensure .sdf extension
            if not fname.lower().endswith('.sdf'):
                fname_out = fname + '.sdf'
            else:
                fname_out = fname
            sdf_path = os.path.join(dest_ligand_dir, fname_out)
            with Chem.SDWriter(sdf_path) as writer:
                writer.write(mol)
        else:
            print(f'Warning: Could not parse mol for {fname}')


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
    synth_csv = os.path.join(results_dir, f'synthesizability_scores_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv')
    lipinski_csv = os.path.join(results_dir, f'lipinski_pass_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv')
    tanimoto_inter_csv = os.path.join(results_dir, f'tanimoto_inter_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv')

    if aurora == 'A':
        aur_type = '4cfg'
    elif aurora == 'B':
        aur_type = '4af3'
    else:
        raise ValueError('Aurora type must be "A" or "B".')
    
    dest_dir = os.path.join(parent_dir, f'docking/{aur_type}/experiment_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}/ligands')

    output_csv = os.path.join(results_dir, f'merged_scores_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv')
    top_100_output_csv = os.path.join(results_dir, f'top_100_tanimoto_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv')
    top_50_output_csv = os.path.join(results_dir, f'top_50_sascore_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv')

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(top_100_output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(top_50_output_csv), exist_ok=True)

    merged_results = merge_on_smiles(synth_path=synth_csv, 
                                     lipinski_path=lipinski_csv,
                                     tanimoto_path=tanimoto_inter_csv,
                                     output_path=output_csv)
                    
    top_100_tanimoto = export_top_100_tanimoto(merged_results)    
    top_50_sa_score = export_top_50_sa_score(top_100_tanimoto)

    top_100_tanimoto.to_csv(top_100_output_csv, index=False)
    top_50_sa_score.to_csv(top_50_output_csv, index=False)

    if epoch != 0:
        # Calculating scores for generated molecules
        mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)

    else:
        # Calculating scores for Aurora inhibitors
        mols, smiles, filenames, fps = read_aurora_kinase_interactions(known_inhib_file)
    
    copy_top_50_ligands(mols,
                        top_50_sa_score=top_50_sa_score,
                        dest_ligand_dir=dest_dir)
    
    print(f'Top 100 Tanimoto results saved to {top_100_output_csv}')
    print(f'Top 50 SA_Score results saved to {top_50_output_csv}')


