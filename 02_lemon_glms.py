import os
import mne
import osl
import glob
import h5py
import sails
import pandas as pd
import glmtools as glm

from anamnesis import obj_from_hdf5file
import matplotlib.pyplot as plt

import numpy as np
from dask.distributed import Client

from lemon_support import (lemon_make_blinks_regressor,
                           lemon_make_task_regressor,
                           lemon_make_bads_regressor,
                           quick_plot_eog_icas,
                           quick_plot_eog_epochs,
                           get_eeg_data,
                           lemon_create_heog,
                           lemon_ica, lemon_check_ica)

from glm_config import cfg

#%% --------------------------------------------------------------
# Functions for first level analysis



def run_first_level(fname, outdir):
    runname = fname.split('/')[-1].split('.')[0]
    print('processing : {0}'.format(runname))

    subj_id = osl.utils.find_run_id(fname)

    raw = mne.io.read_raw_fif(fname, preload=True)

    icaname = fname.replace('preproc_raw.fif', 'ica.fif')
    ica = mne.preprocessing.read_ica(icaname)

    picks = mne.pick_types(raw.info, eeg=True, ref_meg=False)
    chlabels = np.array(raw.info['ch_names'], dtype=h5py.special_dtype(vlen=str))[picks]

    #%% ----------------------------------------------------------
    # GLM-Prep

    # Make blink regressor
    fout = os.path.join(outdir, '{subj_id}_blink-summary.png'.format(subj_id=subj_id))
    blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(raw, figpath=fout)

    fout = os.path.join(outdir, '{subj_id}_icaeog-summary.png'.format(subj_id=subj_id))
    quick_plot_eog_icas(raw, ica, figpath=fout)

    fout = os.path.join(outdir, '{subj_id}_{0}.png'.format('{0}', subj_id=subj_id))
    quick_plot_eog_epochs(raw, figpath=fout)

    veog = raw.get_data(picks='ICA-VEOG')[0, :]**2
    veog = veog > np.percentile(veog, 97.5)

    heog = raw.get_data(picks='ICA-HEOG')[0, :]**2
    heog = heog > np.percentile(heog, 97.5)

    # Make task regressor
    task = lemon_make_task_regressor({'raw': raw})

    # Make bad-segments regressor
    bads_raw = lemon_make_bads_regressor(raw, mode='raw')
    bads_diff = lemon_make_bads_regressor(raw, mode='diff')

    # Get data
    XX = get_eeg_data(raw).T
    print(XX.shape)

    # Run GLM-Periodogram
    conds = {'Eyes Open': task == 1, 'Eyes Closed': task == -1}
    covs = {'Linear Trend': np.linspace(0, 1, raw.n_times)}
    confs = {'Bad Segments': bads_raw,
             'Bad Segments Diff': bads_diff,
             'V-EOG': veog, 'H-EOG': heog}
    conts = [{'name': 'Mean', 'values':{'Eyes Open': 0.5, 'Eyes Closed': 0.5}},
             {'name': 'Open < Closed', 'values':{'Eyes Open': 1, 'Eyes Closed': -1}}]

    fs = raw.info['sfreq']

    # Null model
    freq_vect0, copes0, varcopes0, extras0 = sails.stft.glm_periodogram(XX, axis=0,
                                                                    fit_constant=False,
                                                                    conditions=conds,
                                                                    contrasts=conts,
                                                                    nperseg=int(fs*2),
                                                                    fmin=0.1, fmax=100,
                                                                    fs=fs, mode='magnitude',
                                                                    fit_method='glmtools')
    model0, design0, data0 = extras0
    print(model0.contrast_names)
    print(model0.design_matrix.shape)

    # Full model
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(XX, axis=0,
                                                                    fit_constant=False,
                                                                    conditions=conds,
                                                                    covariates=covs,
                                                                    confounds=confs,
                                                                    contrasts=conts,
                                                                    nperseg=int(fs*2),
                                                                    fmin=0.1, fmax=100,
                                                                    fs=fs, mode='magnitude',
                                                                    fit_method='glmtools')
    model, design, data = extras
    print(model.contrast_names)
    print(model.design_matrix.shape)

    data.info['dim_labels'] = ['Windows', 'Frequencies', 'Sensors']

    print('----')
    print('Null Model AIC : {0} - R2 : {1}'.format(model0.aic.mean(), model0.r_square.mean()))
    print('Full Model AIC : {0} - R2 : {1}'.format(model.aic.mean(), model.r_square.mean()))

    data.info['dim_labels'] = ['Windows', 'Frequencies', 'Sensors']

    hdfname = os.path.join(outdir, '{subj_id}_glm-data.hdf5'.format(subj_id=subj_id))
    if os.path.exists(hdfname):
        print('Overwriting previous results')
        os.remove(hdfname)
    with h5py.File(hdfname, 'w') as F:
        model.to_hdf5(F.create_group('model'))
        model0.to_hdf5(F.create_group('null_model'))
        design.to_hdf5(F.create_group('design'))
        data.to_hdf5(F.create_group('data'))
        F.create_dataset('freq_vect', data=freq_vect)
        F.create_dataset('chlabels', data=chlabels)
        F.create_dataset('scan_duration', data=raw.times[-1])
        F.create_dataset('num_blinks', data=numblinks)

    fout = os.path.join(outdir, '{subj_id}_glm-design.png'.format(subj_id=subj_id))
    design.plot_summary(show=False, savepath=fout)
    fout = os.path.join(outdir, '{subj_id}_glm-efficiency.png'.format(subj_id=subj_id))
    design.plot_efficiency(show=False, savepath=fout)

    quick_plot_firstlevel(hdfname, raw.filenames[0])


def quick_plot_firstlevel(hdfname, rawpath):

    raw = mne.io.read_raw_fif(rawpath).pick_types(eeg=True)

    model = obj_from_hdf5file(hdfname, 'model')	
    freq_vect = h5py.File(hdfname, 'r')['freq_vect'][()]

    tstat_args = {'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 1}
    ts = model.get_tstats(**tstat_args)
    ts = model.copes

    plt.figure(figsize=(16, 9))
    for ii in range(9):
        ind = 6 if ii < 5 else 11
        ax = plt.subplot(4, 5, ii+ind)
        #qlt.plot_joint_spectrum(ax, ts[ii, :, :], raw, xvect=freq_vect, freqs=[1, 9], base=0.5, topo_scale=None)
        ax.plot(freq_vect, ts[ii, :, :])
        ax.set_title(model.contrast_names[ii])

    outf = hdfname.replace('.hdf5', '_glmsummary.png')
    plt.savefig(outf, dpi=300)
    plt.close('all')




#%% ---------------------------------------------------------
# Select datasets to run

# Set to * 'sub-010060' to run single subject in paper examples
#        * 'all' to run everyone
#        * None to run nothing and skip to group preparation
subj = 'all'

n_dask_workers = 4  # Number of dask parallel workers to use if subj is 'all'


if __name__ == '__main__':


    #%% -----------------------------------------------------------
    #  Run first level GLMs - probably should dask-ify this...

    proc_outdir = cfg['lemon_processed_data']

    fbase = os.path.join(cfg['lemon_processed_data'], '{subj}_preproc_raw.fif')
    st = osl.utils.Study(fbase)

    if subj == 'all':
        inputs = st.match_files
    else:
        inputs = st.get(subj=subj)
  
    #client = Client(n_workers=n_dask_workers, threads_per_worker=1)
    #osl.utils.parallel.dask_parallel_bag(run_first_level, inputs, func_args=[cfg['lemon_glm_data']])

    for fname in inputs:
        try:
            run_first_level(fname, cfg['lemon_glm_data'])
        except Exception as e:
            print(e)
            pass


    #%% -------------------------------------------------------
    # Prepare first leve files for group analysis

    print('Loading first levels')

    # Get first level filenames
    fnames = sorted(glob.glob(cfg['lemon_glm_data'] + '/*glm-data.hdf5'))

    # Load subject meta data
    fname = 'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
    meta_file = os.path.join(os.path.dirname(cfg['lemon_raw'].rstrip('/')), fname)
    df = pd.read_csv(meta_file)

    # Extract subject IDs
    allsubj = np.unique([fname.split('/')[-1].split('_')[0][4:] for fname in fnames])
    allsubj_no = np.arange(len(allsubj))

    # Preallocate lists
    subj_id = []
    subj = []
    age = []
    sex = []
    hand = []
    task = []
    scandur = []
    num_blinks = []

    first_level = []
    first_level_null = []
    r2 = []
    r2_null = []
    aic = []
    aic_null = []
    sw = []
    sw_null = []

    # Main loop - load first levels and store copes + meta data
    for idx, fname in enumerate(fnames):
        print('{0}/{1} - {2}'.format(idx, len(fnames), fname.split('/')[-1]))

        # Load data and design
        design = obj_from_hdf5file(fname, 'design')
        data = obj_from_hdf5file(fname, 'data')

        # Load null model
        model = obj_from_hdf5file(fname, 'null_model')
        model.design_matrix = design.design_matrix[:, :2]
        first_level_null.append(model.copes[None, :, :, :])
        r2_null.append(model.r_square.mean())
        sw_null.append(model.get_shapiro(data.data).mean())
        aic_null.append(model.aic.mean())

        # Load full model
        model = obj_from_hdf5file(fname, 'model')
        model.design_matrix = design.design_matrix
        first_level.append(model.copes[None, :, :, :])
        r2.append(model.r_square.mean())
        sw.append(model.get_shapiro(data.data).mean())
        aic.append(model.aic.mean())

        s_id = fname.split('/')[-1].split('_')[0][4:]
        subj.append(np.where(allsubj == s_id)[0][0])
        subj_id.append(s_id)
        if fname.find('EO') > 0:
            task.append(1)
        elif fname.find('EC') > 0:
            task.append(2)

        demo_ind = np.where(df['ID'].str.match('sub-' + s_id))[0]
        if len(demo_ind) > 0:
            tmp_age = df.iloc[demo_ind[0]]['Age']
            age.append(np.array(tmp_age.split('-')).astype(float).mean())
            sex.append(df.iloc[demo_ind[0]]['Gender_ 1=female_2=male'])
        num_blinks.append(h5py.File(fname, 'r')['num_blinks'][()])

    # Stack first levels into new glm dataset +  save
    first_level = np.concatenate(first_level, axis=0)
    group_data = glm.data.TrialGLMData(data=first_level, subj_id=subj_id, shapiro=sw,
                                       subj=subj, task=task, age=age, num_blinks=num_blinks,
                                       sex=sex, scandur=scandur, aic=aic, r2=r2)

    outf = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_groupdata.hdf5')
    with h5py.File(outf, 'w') as F:
        group_data.to_hdf5(F.create_group('data'))
        F.create_dataset('aic', data=aic)
        F.create_dataset('r2', data=r2)

    first_level_null = np.concatenate(first_level_null, axis=0)
    group_data = glm.data.TrialGLMData(data=first_level_null, subj_id=subj_id, shapiro=sw_null,
                                       subj=subj, task=task, age=age, num_blinks=num_blinks,
                                       sex=sex, scandur=scandur, aic=aic_null, r2=r2_null)

    outf = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_groupdata_null.hdf5')
    with h5py.File(outf, 'w') as F:
        group_data.to_hdf5(F.create_group('data'))
        F.create_dataset('aic', data=aic_null)
        F.create_dataset('r2', data=r2_null)