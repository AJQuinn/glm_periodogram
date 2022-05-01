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

extra_funcs = [lemon_set_channel_montage, lemon_create_heog, lemon_ica]

treename = os.path.join('lemon.tree')
lemon = osl.utils.StudyTree(treename, cfg['lemon_raw'])
inputs = sorted(lemon.get('vhdr'))
print(len(inputs))

config = osl.preprocessing.load_config('lemon_preproc.yml')

proc_outdir = '/ohba/pi/knobre/ajquinn/lemon/processed_data/'

#dataset = osl.preprocessing.run_proc_chain(inputs[10], config, outdir=proc_outdir, extra_funcs=extra_funcs)

#goods = osl.preprocessing.run_proc_batch(config, inputs[33:], proc_outdir, overwrite=True, nprocesses=3, extra_funcs=extra_funcs)


#%% ---------------------------------------------------

def get_eeg_data(raw):
    baseref = '/ohba/pi/knobre/datasets/MBB-LEMON/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/sub-010003_EO.set'
    ref = mne.io.read_raw_eeglab(baseref, preload=True)
    mon = ref.get_montage()

    # Get data from same picks
    X = raw.get_data(picks='eeg')

    # Load ideal layout and match data-channels
    raw = raw.copy().pick_types(eeg=True)
    ideal_inds = [mon.ch_names.index(c) for c in raw.info['ch_names']]

    # Preallocate & store ouput
    Y = np.zeros((len(mon.ch_names), X.shape[1]))

    Y[ideal_inds, :] = X

    return Y


def make_eog_regressor(eog):
    bads = sails.utils.detect_artefacts(eog, axis=0, reject_mode='segments', segment_len=2500)
    eog = eog**2
    thresh = np.percentile(eog[bads==False], 95)
    eog = (eog>thresh).astype(float)
    eog[bads] = 0
    return eog


def run_first_level(fname, outdir):
    runname = fname.split('/')[-1].split('.')[0]
    print('processing : {0}'.format(runname))

    subj_id = osl.utils.find_run_id(fname)

    raw = mne.io.read_raw_fif(fname, preload=True)

    picks = mne.pick_types(raw.info, eeg=True, ref_meg=False)
    chlabels = np.array(raw.info['ch_names'], dtype=h5py.special_dtype(vlen=str))[picks]

    #%% ----------------------------------------------------------
    # GLM-Prep

    # Make blink regressor
    fout = os.path.join(outdir, '{subj_id}_blink-summary.png'.format(subj_id=subj_id))
    blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(raw, figpath=fout)
    fout = os.path.join(outdir, '{subj_id}_icaeog-summary.png'.format(subj_id=subj_id))
    quick_plot_eogs(raw, figpath=fout)

    # No reason to square if binarising?
    #veog = raw.get_data(picks='ICA-VEOG')[0, :]**2
    #thresh = np.percentile(veog, 95)
    #veog = (veog>thresh).astype(float)
    veog = make_eog_regressor(raw.get_data(picks='ICA-VEOG')[0, :])
    heog = make_eog_regressor(raw.get_data(picks='ICA-HEOG')[0, :])

    #heog = raw.get_data(picks='ICA-HEOG')[0, :]**2
    #thresh = np.percentile(heog, 95)
    #heog = (heog>thresh).astype(float)

    # Make task regressor
    task = lemon_make_task_regressor({'raw': raw})

    # Make bad-segments regressor
    bads = lemon_make_bads_regressor({'raw': raw})

    # Get data
    XX = get_eeg_data(raw).T
    XX = stats.zscore(XX, axis=0)
    print(XX.shape)

    # Run GLM-Periodogram
    covs = {'Linear Trend': np.linspace(0, 1, raw.n_times),
            'Eyes Open>Closed': task}
    cons = {'Bad Segments': bads, 'V-EOG': veog, 'H-EOG': heog}
    fs = raw.info['sfreq']

    # Null model
    freq_vect0, copes0, varcopes0, extras0 = sails.stft.glm_periodogram(XX, axis=0,
                                                                    nperseg=int(fs*2),
                                                                    fmin=0.1, fmax=100,
                                                                    fs=fs, mode='magnitude',
                                                                    fit_method='glmtools')
    model0, design0, data0 = extras0

    # Full model
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(XX, axis=0,
                                                                    covariates=covs,
                                                                    confounds=cons,
                                                                    nperseg=int(fs*2),
                                                                    fmin=0.1, fmax=100,
                                                                    fs=fs, mode='magnitude',
                                                                    fit_method='glmtools')
    model, design, data = extras

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

    tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 1}
    ts = model.get_tstats(**tstat_args)

    plt.figure(figsize=(16, 9))
    for ii in range(6):
        ind = 4 if ii < 3 else 7
        ax = plt.subplot(4, 3, ii+ind)
        qlt.plot_joint_spectrum(ax, ts[ii, :, :], raw, xvect=freq_vect, freqs=[1, 9], base=0.5, topo_scale=None)

    outf = hdfname.replace('.hdf5', '_glmsummary.png')
    plt.savefig(outf, dpi=300)
    plt.close('all')



proc_outdir = '/ohba/pi/knobre/ajquinn/lemon/processed_data/'
proc_outputs = sorted(glob.glob(os.path.join(proc_outdir, 'sub-*_preproc_raw.fif')))

glm_outdir = '/ohba/pi/knobre/ajquinn/lemon/glm_data/'

eye

for fname in proc_outputs:
    try:
        run_first_level(fname, glm_outdir)
    except Exception as e:
        print(e)
        pass


#%% -------------------------------------------------------

from anamnesis import obj_from_hdf5file

fnames = sorted(glob.glob(glm_outdir + '/*glm-data.hdf5'))
df = pd.read_csv('/ohba/pi/knobre/ajquinn/lemon/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')

allsubj = np.unique([fname.split('/')[-1].split('_')[0][4:] for fname in fnames])
allsubj_no = np.arange(len(allsubj))

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
for idx, fname in enumerate(fnames):
    print('{0}/{1} - {2}'.format(idx, len(fnames), fname.split('/')[-1]))
    model = obj_from_hdf5file(fname, 'null_model')
    design = obj_from_hdf5file(fname, 'design')
    model.design_matrix = design.design_matrix
    first_level_null.append(model.copes[None, :, :, :])
    r2_null.append(model.r_square.mean())
    aic_null.append(model.aic.mean())
    model = obj_from_hdf5file(fname, 'model')
    model.design_matrix = design.design_matrix
    first_level.append(model.copes[None, :, :, :])
    r2.append(model.r_square.mean())
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

first_level = np.concatenate(first_level, axis=0)
group_data = glm.data.TrialGLMData(data=first_level, subj_id=subj_id,
                                   subj=subj, task=task, age=age, num_blinks=num_blinks,
                                   sex=sex, scandur=scandur)

outf = os.path.join(glm_outdir, 'lemon_eeg_sensorglm_groupdata.hdf5')
with h5py.File(outf, 'w') as F:
    group_data.to_hdf5(F.create_group('data'))
    F.create_dataset('aic', data=aic)
    F.create_dataset('r2', data=r2)

first_level_null = np.concatenate(first_level_null, axis=0)
group_data = glm.data.TrialGLMData(data=first_level_null, subj_id=subj_id,
                                   subj=subj, task=task, age=age, num_blinks=num_blinks,
                                   sex=sex, scandur=scandur)

outf = os.path.join(glm_outdir, 'lemon_eeg_sensorglm_groupdata_null.hdf5')
with h5py.File(outf, 'w') as F:
    group_data.to_hdf5(F.create_group('data'))
    F.create_dataset('aic', data=aic_null)
    F.create_dataset('r2', data=r2_null)
