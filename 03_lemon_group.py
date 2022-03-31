import os
import mne
import sys
import osl
import dill
import h5py
import sails
import numpy as np
import pandas as pd
import glmtools as glm
from copy import deepcopy
from anamnesis import obj_from_hdf5file

sys.path.append('/Users/andrew/src/qlt')
import qlt

from glm_config import cfg

#%% ---------------------------------------------------
# Load single subject for reference
from lemon_support import (lemon_set_channel_montage, lemon_ica)

config = osl.preprocessing.load_config('lemon_preproc.yml')

# Drop preproc after montage - only really need the channel info
config['preproc'] = config['preproc'][:2]

base = '/Users/andrew/Projects/lemon/EEG_Raw_BIDS_ID/sub-010060/RSEEG'
infile = os.path.join(base, 'sub-010060.vhdr')
extras = [lemon_set_channel_montage, lemon_ica]
dataset = osl.preprocessing.run_proc_chain(infile, config, extra_funcs=extras)

raw = dataset['raw'].pick_types(eeg=True)

#%% --------------------------------------------------
# Load first level results and fit group model

inputs = os.path.join(cfg['lemon_analysis_dir'], 'lemon_eeg_sensorglm_groupdata.hdf5')

data = obj_from_hdf5file(inputs, 'data')
with h5py.File(inputs, 'r') as F:
    #freq_vect = F['freq_vect'][()]  # needs fixing server-side
    freq_vect = np.linspace(0.5, 100, 200)
    #freq_vect = np.linspace(0.5, 48, 96)

# Drop obvious outliers
bads = sails.utils.detect_artefacts(data.data[:, 0, :, :], axis=0)
clean_data = data.drop(np.where(bads)[0])

# Load age and sex data
df = pd.read_csv('/Users/andrew/Projects/ntad/RA_Interview/LEMON_RA_InterviewData2.csv')
age = []
sex = []
for idx, subj in enumerate(data.info['subj_id']):
    subj = 'sub-' + subj
    ind = np.where(df['ID'] == subj)[0][0]
    row = df.iloc[ind]
    age.append(row['Age'] + 2.5)
    sex.append(row['Gender_ 1=female_2=male'])

data.info['age'] = age
data.info['sex'] = sex
data.info['subj_id'] = list(data.info['subj_id'])

keeps = np.where(np.array(age) < 45)[0][:48]
drops = np.setdiff1d(np.arange(len(age)), keeps)

data = data.drop(drops)

# Refit group model
DC = glm.design.DesignConfig()
DC.add_regressor(name='Mean', rtype='Constant')
DC.add_regressor(name='NumBlinks', rtype='Parametric', datainfo='num_blinks', preproc='z')
DC.add_regressor(name='Sex', rtype='Parametric', datainfo='sex', preproc='z')
DC.add_simple_contrasts()

design = DC.design_from_datainfo(data.info)
gmodel = glm.fit.OLSModel(design, data)

# Housekeeping and rescaling
fl_contrast_names = ['Mean', 'Linear Trend', 'Eyes Open>Closed', 'Bad Segments', 'VEOG', 'HEOG']
gl_contrast_names = gmodel.contrast_names

with h5py.File(os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_glm-data.hdf5'), 'w') as F:
     gmodel.to_hdf5(F.create_group('model'))
     design.to_hdf5(F.create_group('design'))
     #data.to_hdf5(F.create_group('data'))

fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_glm-design.png')
design.plot_summary(show=False, savepath=fout)
fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_glm-efficiency.png')
design.plot_efficiency(show=False, savepath=fout)

#%% ------------------------------------------------------
# Permutation stats - run or load from disk

adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(raw.info, 'eeg')
ntests = np.prod(data.data.shape[2:])
ntimes = data.data.shape[2]
adjacency = mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)

cft = 3
tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt',
              'window_size': 15, 'smooth_dims': 1}

# Permuate
# Blinks on Mean, Mean on linear, task, blinks, bads
P = []

to_permute = [(0, 0, 'Overall Mean'),
              (0, 1, 'Group Mean of Linear Trend'),
              (0, 2, 'Group Mean of Task Effect'),
              (0, 3, 'Group Mean of Bad Segments'),
              (0, 4, 'Group Mean of VEOG'),
              (0, 5, 'Group Mean of HEOG'),
              (1, 0, 'Group Effect of NumBlinks on Mean'),
              (2, 0, 'Group Effect of Sex on Mean')]

run_perms = True
for icon in range(len(to_permute)):
    if run_perms:
        gl_con = to_permute[icon][0]
        fl_con = to_permute[icon][1]
        # Only working with mean regressor for the moment
        fl_mean_data = deepcopy(data)
        fl_mean_data.data = data.data[:, fl_con, : ,:]

        p = glm.permutations.MNEClusterPermutation(design, fl_mean_data, gl_con, 100,
                                                   nprocesses=3,
                                                   metric='tstats',
                                                   cluster_forming_threshold=cft,
                                                   tstat_args=tstat_args,
                                                   adjacency=adjacency)

        with open(os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_perms-con{0}.pkl'.format(icon)), "wb") as dill_file:
            dill.dump(p, dill_file)

        P.append(p)
    else:
        dill_file = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_perms-con{0}.pkl'.format(icon))
        P.append(dill.load(open(dill_file, 'rb')))

#%% ----------------------------

def plot_design(ax, design_matrix, regressor_names):
    num_observations, num_regressors = design_matrix.shape
    vm = np.max((design_matrix.min(), design_matrix.max()))
    cax = ax.pcolormesh(design_matrix, cmap=plt.cm.coolwarm,
                    vmin=-vm, vmax=vm)
    ax.set_xlabel('Regressors')
    tks = np.arange(len(regressor_names)+1)
    ax.set_xticks(tks+0.5)
    ax.set_xticklabels(tks)

    tkstep = 2
    tks = np.arange(0, design_matrix.shape[0], tkstep)

    for tag in ['top', 'right', 'left', 'bottom']:
        ax.spines[tag].set_visible(False)

    summary_lines = True
    new_cols = 0
    for ii in range(num_regressors):
        if summary_lines:
            x = design_matrix[:, ii]
            if np.abs(np.diff(x)).sum() != 0:
                #rn = np.max((np.abs(x.max()), np.abs(x.min())))
                #if (x.max() > 0) and (x.min() < 0):
                #    rn = rn*2
                #y = (x-x.min()) / (rn) * .8 + .1
                #y = (x) / (rn) * .8 + .1
                y = (0.5*x) / (np.max(np.abs(x)) * 1.1)
            else:
                # Constant regressor
                y = np.ones_like(x) * .45
            if num_observations > 50:
                ax.plot(y+ii+new_cols+0.5, np.arange(0, 0+num_observations)+.5, 'k')
            else:
                yy = y+ii+new_cols+0.5
                print('{} - {} - {}'.format(yy.min(), yy.mean(), yy.max()))
                ax.plot(y+ii+new_cols+0.5, np.arange(0, 0+num_observations)+.5,
                        'k|', markersize=5)

        # Add white dividing line
        if ii < num_regressors-1:
            ax.plot([ii+1+new_cols, ii+1+new_cols], [0, 0+num_observations],
                    'w', linewidth=4)
    return cax

I = np.argsort(data.data[:, 0, :, 23].sum(axis=1))
I = np.arange(48)

plt.figure(figsize=(16, 9))
aspect = 16/9
xf = qlt.prep_scaled_freq(0.5, freq_vect)

subj_ax = plt.axes([0.05, 0.225, 0.35, 0.35*aspect])
des_ax = plt.axes([0.425, 0.225, 0.175, 0.51])
cb_dm = plt.axes([0.62, 0.225, 0.01, 0.2])
mean_ax = plt.axes([0.75, 0.6, 0.23, 0.25])
cov_ax = plt.axes([0.75, 0.1, 0.23, 0.25])

subj_ax.plot((0, 35*len(I)), (0, 2e-4*len(I)), color=[0.8, 0.8, 0.8], lw=0.5)
for ii in range(48):
    d = data.data[I[ii],0, :, 23]
    subj_ax.plot(np.arange(len(freq_vect))+35*ii, d + 2e-4*ii)
for tag in ['top','right']:
    subj_ax.spines[tag].set_visible(False)
subj_ax.spines['bottom'].set_bounds(0, len(freq_vect))
subj_ax.spines['left'].set_bounds(0, 0.1e-2)
subj_ax.set_xlim(0)
subj_ax.set_ylim(0)
subj_ax.set_xticks([])
subj_ax.set_yticks([])
l = subj_ax.set_xlabel(r'Frequency (Hz) $\rightarrow$', loc='left')
l = subj_ax.set_ylabel(r'Amplitude $\rightarrow$', loc='bottom')
subj_ax.text(48+35*19, 2e-4*12, r'Participants $\rightarrow$', rotation=45)
subj_ax.set_title('First level GLM Spectra')
qlt.subpanel_label(subj_ax, chr(65), yf=1.05, xf=0.25)


pcm = plot_design(des_ax, design.design_matrix[I, :], design.regressor_names)
des_ax.spines['left'].set_bounds(0, 24)
des_ax.set_yticks(np.arange(10)*5)
des_ax.set_xticklabels(['Mean', 'Num Blinks', 'Sex', ''])
des_ax.set_ylabel('Participants')
des_ax.set_title('Group Design Matrix')
qlt.subpanel_label(des_ax, chr(65+1), yf=1.05)
plt.colorbar(pcm, cax=cb_dm)


mean_ax.errorbar(xf[0], gmodel.copes[0, 0, :, 23], yerr=np.sqrt(gmodel.varcopes[0, 0, :, 23]), errorevery=1)
mean_ax.set_xticks(xf[2], xf[1])
mean_ax.set_title('Group Mean Spectrum')
qlt.decorate_spectrum(mean_ax, ylabel='Amplitude')
qlt.subpanel_label(mean_ax, chr(65+2), yf=1.1)
mean_ax.set_xlim(0)
mean_ax.set_ylim(0)

cov_ax.errorbar(xf[0], gmodel.copes[1, 0, :, 23], yerr=np.sqrt(gmodel.varcopes[1, 0, :, 23]), errorevery=2)
cov_ax.set_title('Group effects on Mean Spectrum')
cov_ax.errorbar(xf[0], gmodel.copes[2, 4, :, 23], yerr=np.sqrt(gmodel.varcopes[1, 4, :, 23]), errorevery=2)
cov_ax.set_xticks(xf[2], xf[1])
cov_ax.legend(['Blinks', 'Sex'])
qlt.decorate_spectrum(cov_ax, ylabel='Amplitude')
qlt.subpanel_label(cov_ax, chr(65+3), yf=1.1)


fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_glm-overview.png')
plt.savefig(fout, transparent=True, dpi=300)

#%% ----------------------------

plt.figure(figsize=(16, 9))

ax = plt.axes([0.05, 0.45, 0.2, 0.3])
qlt.plot_joint_spectrum(ax, gmodel.copes[0, 0, :, :],
                        raw, freq_vect, base=0.5,
                        freqs=[1, 9, 20], topo_scale=None)
qlt.subpanel_label(ax, chr(65), yf=1.1)

ax = plt.axes([0.075, 0.05, 0.15, 0.225])
qlt.plot_channel_layout(ax, raw)
qlt.subpanel_label(ax, chr(65+1), yf=1.1)

for ii in range(4):
    ax = plt.axes([0.325+0.165*ii, 0.6, 0.133, 0.2])
    #qlt.plot_joint_spectrum(ax, gmodel.get_tstats(sigma_hat='auto')[0, ii+1, :, :],
    #                        raw, freq_vect, base=0.5,
    #                        freqs=[1, 9, 20], topo_scale=None)
    # Only working with mean regressor for the moment
    fl_mean_data = deepcopy(data)
    fl_mean_data.data = data.data[:, ii+1, : ,:]
    qlt.plot_sensorspace_clusters(fl_mean_data, P[ii+1], raw, ax, base=0.5, title=to_permute[ii+1][2], ylabel='t-stat', thresh=99, xvect=freq_vect)
    qlt.subpanel_label(ax, chr(65+ii+2), yf=1.1)

fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 0, : ,:]
for ii in range(2):
    ax = plt.axes([0.425+0.165*ii*1.5, 0.1, 0.133, 0.2])
    qlt.plot_sensorspace_clusters(fl_mean_data, P[ii+5], raw, ax, base=0.5, title=to_permute[ii+5][2], ylabel='t-stat', xvect=freq_vect)
    qlt.subpanel_label(ax, chr(65+ii+6), yf=1.1)

fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_glm-clusters.png')
plt.savefig(fout, transparent=True, dpi=300)

#%% ----------------------------
# Sanity check figure

plt.figure()
for ii in range(3):
    for jj in range(6):
        ind = (jj+1)+ii*6
        print(ind)
        plt.subplot(3, 6, ind)

        ts = gmodel.get_tstats(**tstat_args)[ii, jj, : ,:]

        plt.plot(freq_vect, ts)
        plt.title(gl_contrast_names[ii] + ' : ' + fl_contrast_names[jj])

#%% ----------------------------

fx = qlt.prep_scaled_freq(0.5, freq_vect,)

plt.figure(figsize=(16, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.35)
plt.subplots_adjust(hspace=0.4, wspace=0.3, top=1, left=0.075, right=0.975)
ii = 0
for jj in range(6):
    ind = (jj+7)+ii*6
    ax = plt.subplot(3,6,ind)

    if jj == 0:
        qlt.plot_joint_spectrum(ax, gmodel.copes[0, 0, :, :], raw, freq_vect, base=0.5, freqs=[0.5, 9, 24], topo_scale=None)
    else:
        fl_mean_data = deepcopy(data)
        fl_mean_data.data = data.data[:, jj, : ,:]
        qlt.plot_sensorspace_clusters(fl_mean_data, P[jj], raw, ax,
                                      base=0.5, title=to_permute[jj][2],
                                      ylabel='t-stat', thresh=99, xvect=freq_vect)
    qlt.subpanel_label(ax, chr(65+jj), yf=1.1)


    ax = plt.subplot(3,6,ind+6)
    ax.plot(fx[0], fl_mean_data.data.mean(axis=2).T)
    ax.set_xticks(fx[2], fx[1])
    if jj == 0:
        qlt.decorate_spectrum(ax, ylabel='amplitude')
    else:
        qlt.decorate_spectrum(ax, ylabel='')
        ax.set_ylim(-0.002, 0.002)
        if jj > 1:
            ax.set_yticklabels([])

fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_glm-simpleeffects.png')
plt.savefig(fout, transparent=True, dpi=300)


#%% ----------------------------

plt.figure(figsize=(12,8))

ax = plt.subplot(1,3,1)
qlt.plot_joint_spectrum(ax, gmodel.copes[0, 0, :, :], raw, freq_vect,
                        title='Group Mean', base=0.5,
                        freqs=[0.5, 9, 24], topo_scale=None)
qlt.subpanel_label(ax, chr(65), yf=1.1)

fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 0, : ,:]
ax = plt.subplot(1,3,2)
qlt.plot_sensorspace_clusters(fl_mean_data, P[6], raw, ax,
                              base=0.5, title=to_permute[6][2],
                              ylabel='t-stat', thresh=99, xvect=freq_vect)
qlt.subpanel_label(ax, chr(66), yf=1.1)
ax = plt.subplot(1,3,3)
qlt.plot_sensorspace_clusters(fl_mean_data, P[7], raw, ax,
                              base=0.5, title=to_permute[7][2],
                              ylabel='t-stat', thresh=99, xvect=freq_vect)
qlt.subpanel_label(ax, chr(67), yf=1.1)
plt.subplots_adjust(top=0.6)

fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_glm-higherordereffects.png')
plt.savefig(fout, transparent=True, dpi=300)
