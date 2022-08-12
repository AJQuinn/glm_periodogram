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
cfg['lemon_raw_eeg'] = os.path.join(cfg['lemon_raw'], 'EEG_Raw_BIDS_ID')
cfg['lemon_processed_data'] = os.path.join(cfg['lemon_output'], 'preprocessed_data')
cfg['lemon_glm_data'] = os.path.join(cfg['lemon_output'], 'glm_data')
