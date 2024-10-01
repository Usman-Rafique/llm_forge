import os
import yaml


def load_config(config_file, model_name=None):
    """
    Load configuration from a YAML file.
    
    Args:
    config_file (str): Path to the configuration file or name of the config (without .yaml extension)
    model_name (str, optional): Name of the specific model configuration to load
    
    Returns:
    dict or tuple: If model_name is provided, returns (global_config, model_config),
                   otherwise returns the entire config dict
    """
    # If config_file doesn't have a .yaml extension, assume it's a config name in the configs directory
    if not config_file.endswith('.yaml'):
        config_file = os.path.join('configs', f'{config_file}.yaml')

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if model_name:
        global_config = config['global']
        model_config = config['models'][model_name]
        return global_config, model_config
    else:
        return config
