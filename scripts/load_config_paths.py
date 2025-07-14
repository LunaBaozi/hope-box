import yaml
from pathlib import Path


def get_pipeline_config():
    """Load pipeline configuration"""
    # Try to find config file from script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # Go up from external/hope-box/scripts to project root
    config_path = project_root / 'config' / 'config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}")
        return None


def get_graphbp_sdf_path(config, epoch, num_gen, known_binding_site, pdbid):
    """Build GraphBP SDF folder path from config"""
    if not config:
        # Fallback to hardcoded path
        script_dir = Path(__file__).parent
        parent_dir = script_dir.parent.parent  # Go up to external/
        return str(parent_dir / f'graphbp/OpenMI/GraphBP/GraphBP/trained_model_reduced_dataset_100_epochs/gen_mols_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}/sdf')
    
    graphbp_config = config['modules']['graphbp']
    pattern = graphbp_config['output_pattern']
    
    relative_path = pattern.format(
        epoch=epoch,
        num_gen=num_gen,
        known_binding_site=known_binding_site,
        pdbid=pdbid
    )
    
    # Build full path from project root
    project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
    full_path = (project_root / 
                graphbp_config['path'] / 
                graphbp_config['trained_model'] / 
                relative_path)
    
    return str(full_path)


def get_aurora_data_path(config, aurora):
    """Build Aurora kinase data file path from config"""
    script_dir = Path(__file__).parent
    hope_box_dir = script_dir.parent  # Go up to external/hope-box/
    
    if config:
        hope_box_config = config['modules']['hope_box']
        data_dir = hope_box_config['data_dir']
    else:
        data_dir = 'data'
    
    return str(hope_box_dir / data_dir / f'aurora_kinase_{aurora}_interactions.csv')


def get_output_path(config, experiment, epoch, num_gen, known_binding_site, pdbid, output_file_arg=None, filename_prefix='synthesizability_scores'):
    """Determine output file path"""
    if output_file_arg:
        # Use provided output file path (from Snakemake)
        return output_file_arg
    
    # Default behavior for standalone usage
    script_dir = Path(__file__).parent
    hope_box_dir = script_dir.parent  # Go up to external/hope-box/
    
    if config:
        # Use config-based structure
        results_subdir = f'experiment_{experiment}_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}'
        results_dir = hope_box_dir / config['modules']['hope_box']['results_dir'] / results_subdir
    else:
        # Fallback structure
        results_dir = hope_box_dir / 'results' / f'experiment_{experiment}_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}'
    
    # Dynamic filename generation
    output_file = results_dir / f'{filename_prefix}.csv'
    
    # Create directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return str(output_file)



def get_hope_box_results_path(config, experiment, epoch, num_gen, known_binding_site, pdbid, filename):
    """Build HOPE-Box results file path from config"""
    script_dir = Path(__file__).parent
    hope_box_dir = script_dir.parent  # Go up to external/hope-box/
    
    if config:
        results_dir_name = config['modules']['hope_box']['results_dir']
    else:
        results_dir_name = 'results'
    
    results_subdir = f'experiment_{experiment}_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}'
    full_path = hope_box_dir / results_dir_name / results_subdir / filename
    
    # Create directory if it doesn't exist
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    return str(full_path)


def get_project_root():
    """Get the project root directory"""
    script_dir = Path(__file__).parent
    return script_dir.parent.parent.parent  # Go up from external/hope-box/scripts to project root


def get_module_path(config, module_name):
    """Get the path for a specific module"""
    if not config:
        project_root = get_project_root()
        return str(project_root / 'external' / module_name)
    
    module_config = config['modules'].get(module_name, {})
    if 'path' in module_config:
        project_root = get_project_root()
        return str(project_root / module_config['path'])
    
    # Fallback
    project_root = get_project_root()
    return str(project_root / 'external' / module_name)


class PipelinePaths:
    """Path management class for the drug design pipeline"""
    
    def __init__(self, config=None):
        self.config = config or get_pipeline_config()
        self.project_root = get_project_root()
    
    def graphbp_sdf_path(self, epoch, num_gen, known_binding_site, pdbid):
        """Get GraphBP SDF folder path"""
        return get_graphbp_sdf_path(self.config, epoch, num_gen, known_binding_site, pdbid)
    
    def aurora_data_path(self, aurora):
        """Get Aurora kinase data file path"""
        return get_aurora_data_path(self.config, aurora)
    
    def output_path(self, experiment, epoch, num_gen, known_binding_site, pdbid, output_file_arg=None, filename_prefix='synthesizability_scores'):
        """Get output file path with dynamic filename"""
        return get_output_path(self.config, experiment, epoch, num_gen, known_binding_site, pdbid, output_file_arg, filename_prefix)
    
    def hope_box_results_path(self, experiment, epoch, num_gen, known_binding_site, pdbid, filename):
        """Get HOPE-Box results file path"""
        return get_hope_box_results_path(self.config, experiment, epoch, num_gen, known_binding_site, pdbid, filename)
    
    def module_path(self, module_name):
        """Get module path"""
        return get_module_path(self.config, module_name)
    
    # Convenience methods for specific file types
    def synthesizability_output_path(self, experiment, epoch, num_gen, known_binding_site, pdbid, output_file_arg=None):
        """Get synthesizability scores output path"""
        return self.output_path(experiment, epoch, num_gen, known_binding_site, pdbid, output_file_arg, 'synthesizability_scores')
    
    def lipinski_output_path(self, experiment, epoch, num_gen, known_binding_site, pdbid, output_file_arg=None):
        """Get Lipinski results output path"""
        return self.output_path(experiment, epoch, num_gen, known_binding_site, pdbid, output_file_arg, 'lipinski_pass')
    
    def admet_output_path(self, epoch, num_gen, known_binding_site, pdbid, output_file_arg=None):
        """Get ADMET results output path"""
        return self.output_path(epoch, num_gen, known_binding_site, pdbid, output_file_arg, 'admet_scores')
    
    def docking_output_path(self, epoch, num_gen, known_binding_site, pdbid, output_file_arg=None):
        """Get docking results output path"""
        return self.output_path(epoch, num_gen, known_binding_site, pdbid, output_file_arg, 'docking_scores')
    
    def equibind_ligands_path(self, experiment, epoch, num_gen, known_binding_site, pdbid):
        """Get EquiBind ligands directory path"""
        experiment_name = f"experiment_{experiment}_{epoch}_{num_gen}_{known_binding_site}_{pdbid}"
        equibind_path = self.project_root / "external" / "equibind" / "data" / pdbid / experiment_name / "ligands"
        return str(equibind_path)