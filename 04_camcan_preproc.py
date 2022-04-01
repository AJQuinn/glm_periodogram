import os
import mne
import osl
import glob
import sails

import numpy as np
import h5py
import pandas as pd
from scipy import stats
import glmtools as glm

from lemon_support import (lemon_make_blinks_regressor,
                           lemon_make_task_regressor,
                           lemon_make_bads_regressor,
                           lemon_set_channel_montage,
                           camcan_ica, lemon_check_ica)
from glm_config import cfg
extra_funcs = [camcan_ica]


treename = os.path.join('camcan.tree')
camcan = osl.utils.StudyTree(treename, cfg['camcan_raw'])
inputs = sorted(camcan.get('trans'))
print(len(inputs))

config = osl.preprocessing.load_config('camcan_preproc.yml')

dataset = osl.preprocessing.run_proc_chain(inputs[0], config, outdir=cfg['camcan_preprocessed_data_dir'], extra_funcs=extra_funcs)

#goods = osl.preprocessing.run_proc_batch(config, inputs[33:], proc_outdir, overwrite=True, nprocesses=3, extra_funcs=extra_funcs)
