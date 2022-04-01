import os
import yaml
import pathlib

code_dir = str(pathlib.Path(__file__).parent.resolve())

yaml_path = os.path.join(code_dir, 'glm_config.yaml')
with open(yaml_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg['yaml_path'] = yaml_path
cfg['code_dir'] = code_dir

# Check LEMON
cfg['lemon_analysis_dir'] = os.path.join(cfg['lemon_output_dir'], 'analysis')
cfg['lemon_preprocessed_data_dir'] = os.path.join(cfg['lemon_output_dir'], 'preprocessed_data')
cfg['lemon_glm_data_dir'] = os.path.join(cfg['lemon_output_dir'], 'glm_data')

cfg['camcan_analysis_dir'] = os.path.join(cfg['camcan_output_dir'], 'analysis')
cfg['camcan_preprocessed_data_dir'] = os.path.join(cfg['camcan_output_dir'], 'preprocessed_data')
cfg['camcan_glm_data_dir'] = os.path.join(cfg['camcan_output_dir'], 'glm_data')
