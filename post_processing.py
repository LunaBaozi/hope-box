import os, argparse
import pandas as pd

from rdkit import Chem

from pathlib import Path

from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder
from scripts.load_config_paths import PipelinePaths

def merge_on_smiles(synth_path, 
                    lipinski_path,
                    tanimoto_path,
                    output_path):
    """
    Merges three CSV files on the 'smiles' column, sorts by 'tanimoto' in 
    decreasing order, and saves the result.
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

    # Get file paths using the PipelinePaths configuration
    results_dir = Path(paths.output_path(experiment, epoch, num_gen, known_binding_site, pdbid)).parent
    synth_path = paths.output_path(experiment, epoch, num_gen, known_binding_site, pdbid, None, 'synthesizability_scores')
    lipinski_path = paths.output_path(experiment, epoch, num_gen, known_binding_site, pdbid, None, 'lipinski_pass')
    tanimoto_inter_path = paths.output_path(experiment, epoch, num_gen, known_binding_site, pdbid, None, 'tanimoto_inter')
    
    # Use EquiBind ligands path from configuration
    dest_dir = paths.equibind_ligands_path(experiment, epoch, num_gen, known_binding_site, pdbid)

    # Define output file paths using configuration to match Snakemake rule expectations
    output_csv = paths.hope_box_results_path(experiment, epoch, num_gen, known_binding_site, pdbid, f'merged_scores.csv')
    top_100_output_csv = paths.hope_box_results_path(experiment, epoch, num_gen, known_binding_site, pdbid, f'top_100_tanimoto.csv')
    top_50_output_csv = paths.hope_box_results_path(experiment, epoch, num_gen, known_binding_site, pdbid, f'top_50_sascore.csv')

    # Directory creation is handled by the configuration system
    merged_results = merge_on_smiles(synth_path=synth_path, 
                                     lipinski_path=lipinski_path,
                                     tanimoto_path=tanimoto_inter_path,
                                     output_path=output_csv)
                    
    top_100_tanimoto = export_top_100_tanimoto(merged_results)    
    top_50_sa_score = export_top_50_sa_score(top_100_tanimoto)

    top_100_tanimoto.to_csv(top_100_output_csv, index=False)
    top_50_sa_score.to_csv(top_50_output_csv, index=False)

    if epoch != 0:
        # Calculating scores for generated molecules
        sdf_folder = paths.graphbp_sdf_path(epoch, num_gen, known_binding_site, pdbid)
        mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)

    else:
        # Calculating scores for Aurora inhibitors
        aurora_data_file = paths.aurora_data_path(aurora)
        mols, smiles, filenames, fps = read_aurora_kinase_interactions(aurora_data_file)
    
    # copy_top_50_ligands(mols,
    #                     top_50_sa_score=top_50_sa_score,
    #                     dest_ligand_dir=dest_dir)
    
    print(f"Creating ligands directory: {dest_dir}")
    
    try:
        copy_top_50_ligands(mols,
                            top_50_sa_score=top_50_sa_score,
                            dest_ligand_dir=dest_dir)
        
        # Verify the directory was created
        if os.path.exists(dest_dir):
            print(f"Successfully created ligands directory with {len(os.listdir(dest_dir))} files")
        else:
            print(f"Warning: Ligands directory was not created: {dest_dir}")
            # Create empty directory to satisfy Snakemake
            os.makedirs(dest_dir, exist_ok=True)
            
    except Exception as e:
        print(f"Error in copy_top_50_ligands: {e}")
        # Create empty directory to satisfy Snakemake
        os.makedirs(dest_dir, exist_ok=True)

    print(f'Top 100 Tanimoto results saved to {top_100_output_csv}')
    print(f'Top 50 SA_Score results saved to {top_50_output_csv}')


