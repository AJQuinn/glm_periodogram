import os
import mne
import osl
import glob
import sails

import numpy as np
import h5py
import pandas as pd
from scipy import stats
from anamnesis import obj_from_hdf5file
import matplotlib.pyplot as plt


from lemon_support import (lemon_make_blinks_regressor,
                           lemon_make_task_regressor,
                           lemon_make_bads_regressor,
                           lemon_set_channel_montage,
                           lemon_create_heog,
                           lemon_ica, lemon_check_ica)
from glm_config import cfg

import sys
sys.path.insert(0, '/home/ajquinn/src/glm')
import glmtools as glm

sys.path.append('/home/ajquinn/src/qlt/')
import qlt

#%% ---------------------------
#
# header errors (wrong DataFile and MarkerFile) in:
# sub-010193.vhdr
# sub-010219.vhdr
#
# Can be fixed by hand

if __name__ == '__main__':

    extra_funcs = [lemon_set_channel_montage, lemon_create_heog, lemon_ica]

    fbase = '/ohba/pi/knobre/datasets/MBB-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/{subj}/RSEEG/{subj}.vhdr'
    inputs = st.match_files
    print(len(inputs))

    config = osl.preprocessing.load_config('lemon_preproc.yml')

    proc_outdir = '/ohba/pi/knobre/ajquinn/lemon/processed_data/'

    #dataset = osl.preprocessing.run_proc_chain(inputs[10], config, outdir=proc_outdir, extra_funcs=extra_funcs)

    goods = osl.preprocessing.run_proc_batch(config, inputs[33:], proc_outdir, overwrite=True, nprocesses=3, extra_funcs=extra_funcs)

