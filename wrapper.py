'''
This script serves as the main entry point for the Aurora kinase CADD pipeline. 
It provides a unified interface to launch the entire workflow, including molecule 
generation, evaluation, post-hoc filtering, and docking, using a single command. 
The wrapper handles argument parsing, manages execution order, and records timing 
for each pipeline stage. 
Designed for flexibility and reproducibility, it simplifies running complex 
analyses by coordinating all necessary scripts and dependencies.

Example usage:
    python wrapper.py --num_gen 100 --epoch 5 --known_binding_site True --aurora B
'''

import argparse
import subprocess
import os, sys, time


def run_script(script_path, args=None):
    # Determine the correct executable
    if script_path.endswith('.sh'):
        cmd = ['bash', script_path]
    elif script_path.endswith('.py'):
        cmd = [sys.executable, script_path]
    else:
        raise ValueError('Unsupported script type. Only .sh and .py are supported.')
    if args:
        cmd += args
    subprocess.run(cmd, check=True)

def run_bash_script_in_conda(script_path, args, conda_env):
    arg_str = ' '.join(args) if args else ''
    command = (
        f'source ~/../../vol/data/miniconda3/etc/profile.d/conda.sh && '
        f'conda activate {conda_env} && '
        f'bash {script_path} {arg_str}'
    )
    subprocess.run(['bash', '-c', command], check=True)


def main():
    parser = argparse.ArgumentParser(description='Wrapper for CADD pipeline targeting Aurora protein kinases.')
    parser.add_argument('--num_gen', type=int, required=False, default=0, help='Desired number of generated molecules (int, positive)')
    parser.add_argument('--epoch', type=int, required=False, default=0, help='Epoch number the model will use to generate molecules (int, 0-99)')
    parser.add_argument('--known_binding_site', type=str, required=False, default='0', help='Allow model to use binding site information (True, False)')
    parser.add_argument('--aurora', type=str, required=False, default='B', help='Aurora kinase type (str, A, B)')
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_gen_path = os.path.join(base_dir, 'main_gen.py')
    main_eval_path = os.path.join(base_dir, 'main_eval.py')
    post_hoc_dir = os.path.join(base_dir, 'post_hoc_filtering')
    docking_dir = os.path.join(base_dir, 'docking')
    
    bash_scripts = [
    # os.path.join(docking_dir, 'run_pipeline.sh'),
    ]

    python_scripts = [
        os.path.join(post_hoc_dir, 'synthesizability_scores.py'),
        os.path.join(post_hoc_dir, 'lipinski.py'),
        os.path.join(post_hoc_dir, 'tanimoto_intra.py'),
        os.path.join(post_hoc_dir, 'tanimoto_inter.py'),
        os.path.join(post_hoc_dir, 'graphics.py'),
        # os.path.join(post_hoc_dir, 'post_processing.py'),
    ]

    bash_param_args = [
        str(args.num_gen),
        str(args.epoch),
        str(args.known_binding_site),
        str(args.aurora).upper()
    ]

    python_param_args = [
        '--num_gen', str(args.num_gen),
        '--epoch', str(args.epoch),
        '--known_binding_site', str(args.known_binding_site),
        '--aurora', str(args.aurora).upper()
    ]

    # Run analyses and measure execution time

    # GENERATION ###############################################################
    start_gen = time.time()
    # run_script(main_gen_path, python_param_args)
    gen_time = time.time() - start_gen

    # EVALUATION ###############################################################
    start_eval = time.time()
    # run_script(main_eval_path, python_param_args)
    eval_time = time.time() - start_eval

    # POST-HOC FILTERING #######################################################
    results_dir = os.path.join(
        base_dir,
        'post_hoc_filtering',
        f'results_epoch_{args.epoch}_mols_{args.num_gen}_bs_{args.known_binding_site}_aurora_{args.aurora}'
    )
    os.makedirs(results_dir, exist_ok=True)
    
    start_filtering = time.time()
    for script in python_scripts:
        run_script(script, python_param_args)
    filtering_time = time.time() - start_filtering

    # DOCKING ##################################################################
    start_docking = time.time()
    for script in bash_scripts:
        run_bash_script_in_conda(script, bash_param_args, conda_env='vina')

    # run_script(os.path.join(docking_dir, 'top_scoring_docking.py'), python_param_args)
    docking_time = time.time() - start_docking
    
    end_time = time.time() - start_gen

    with open(os.path.join(results_dir, f'elapsed_time_{args.epoch}_{args.num_gen}_{args.known_binding_site}_{args.aurora}.txt'), 'w') as f:
        f.write(f'Whole pipeline executed in: ' + time.strftime('%H:%M:%S', time.gmtime(end_time)) + '\n')  
        f.write(f'Generation executed in: ' + time.strftime('%H:%M:%S', time.gmtime(gen_time)) + '\n')
        f.write(f'Evaluation executed in: ' + time.strftime('%H:%M:%S', time.gmtime(eval_time)) + '\n')
        f.write(f'Post-hoc filtering executed in: ' + time.strftime('%H:%M:%S', time.gmtime(filtering_time)) + '\n')
        f.write(f'Docking executed in: ' + time.strftime('%H:%M:%S', time.gmtime(docking_time)) + '\n')

if __name__ == '__main__':
    main()