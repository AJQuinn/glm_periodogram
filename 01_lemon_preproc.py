import os
import mne
import osl
import glob
import sails

import numpy as np
from dask.distributed import Client

from lemon_support import (lemon_make_blinks_regressor,
                           lemon_make_task_regressor,
                           lemon_make_bads_regressor,
                           lemon_set_channel_montage,
                           lemon_create_heog,
                           lemon_ica, lemon_check_ica)

from glm_config import cfg


#%% ---------------------------
#
# header errors (wrong DataFile and MarkerFile) in:
# sub-010193.vhdr
# sub-010219.vhdr
#
# Can be fixed by hand

if __name__ == '__main__':

    extra_funcs = [lemon_set_channel_montage, lemon_create_heog, lemon_ica]

    datadir = '/rds/projects/q/quinna-spectral-changes-in-ageing/raw_data'
    fbase = os.path.join(cfg['lemon_raw'], '{subj}', 'RSEEG', '{subj}.vhdr')

    st = osl.utils.Study(fbase)
    inputs = st.match_files
    print(len(inputs))

    config = osl.preprocessing.load_config('lemon_preproc.yml')

    proc_outdir = '/ohba/pi/knobre/ajquinn/lemon/processed_data/'
    proc_outdir = '/rds/projects/q/quinna-spectral-changes-in-ageing/processed_data'
    proc_outdir = cfg['lemon_processed_data']

    #dataset = osl.preprocessing.run_proc_chain(inputs[10], config, outdir=proc_outdir, extra_funcs=extra_funcs)

    client = Client(n_workers=8, threads_per_worker=1)
    goods = osl.preprocessing.run_proc_batch(config, inputs, proc_outdir, overwrite=True, extra_funcs=extra_funcs, dask_client=True)

