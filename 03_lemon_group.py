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

outdir = '/Users/andrew/Projects/glm/glm_psd/analysis'
figbase = '/Users/andrew/Projects/glm/glm_psd/figures/'

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

inputs = '/Users/andrew/Projects/glm/glm_psd/analysis/group_sensorglm.hdf5'

data = obj_from_hdf5file(inputs, 'data')
with h5py.File(inputs, 'r') as F:
    #freq_vect = F['freq_vect'][()]  # needs fixing server-side
    freq_vect = np.linspace(0.5, 48, 96)

# Drop obvious outliers
bads = sails.utils.detect_artefacts(data.data[:, 0, :, :], axis=0)
clean_data = data.drop(np.where(bads)[0])

# Load age and sex data
df = pd.read_csv('/Users/andrew/Projects/ntad/RA_Interview/LEMON_RA_InterviewData2.csv')
age = []
sex = []
for idx, subj in enumerate(data.info['subjs']):
    ind = np.where(df['ID'] == subj)[0][0]
    row = df.iloc[ind]
    age.append(row['Age'] + 2.5)
    sex.append(row['Gender_ 1=female_2=male'])

data.info['age'] = age
data.info['sex'] = sex
data.info['subjs'] = list(data.info['subjs'])

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
fl_contrast_names = ['Mean', 'linear', 'task', 'badsegments', 'blinks']
gl_contrast_names = gmodel.contrast_names

with h5py.File(os.path.join(outdir, 'lemon-group_glm-data.hdf5'), 'w') as F:
     gmodel.to_hdf5(F.create_group('model'))
     design.to_hdf5(F.create_group('design'))
     #data.to_hdf5(F.create_group('data'))

fout = os.path.join(outdir, 'lemon-group_glm-design.png')
design.plot_summary(show=False, savepath=fout)
fout = os.path.join(outdir, 'lemon-group_glm-efficiency.png')
design.plot_efficiency(show=False, savepath=fout)

#%% ------------------------------------------------------
# Permutation stats - run or load from disk

adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(raw.info, 'eeg')
ntests = np.prod(data.data.shape[2:])
ntimes = data.data.shape[2]
adjacency = mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)

cft = 2.3
tstat_args = {'sigma_hat': 'auto'}

# Permuate
# Blinks on Mean, Mean on linear, task, blinks, bads
P = []

to_permute = [(0, 0, 'Overall Mean'),
              (0, 1, 'Group Mean of Linear Trend'),
              (0, 2, 'Group Mean of Task Effect'),
              (0, 3, 'Group Mean of Bad Segments'),
              (0, 4, 'Group Mean of Blinks'),
              (1, 0, 'Group Effect of Blinks on Mean'),
              (2, 0, 'Group Effect of Sex on Mean')]

run_perms = False
for icon in range(len(to_permute)):
    if run_perms:
        gl_con = to_permute[icon][0]
        fl_con = to_permute[icon][1]
        # Only working with mean regressor for the moment
        fl_mean_data = deepcopy(data)
        fl_mean_data.data = data.data[:, fl_con, : ,:]

        p = glm.permutations.MNEClusterPermutation(design, fl_mean_data, gl_con, 1000,
                                                   nprocesses=3,
                                                   metric='tstats',
                                                   cluster_forming_threshold=cft,
                                                   tstat_args=tstat_args,
                                                   adjacency=adjacency)

        with open(os.path.join(outdir, 'lemon-group_perms-con{0}.pkl'.format(icon)), "wb") as dill_file:
            dill.dump(p, dill_file)

        P.append(p)
    else:
        dill_file = os.path.join(outdir, 'lemon-group_perms-con{0}.pkl'.format(icon))
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

plt.figure(figsize=(16, 9))

subj_ax = plt.axes([0.08, 0.225, 0.2279411764705882, 0.77])
des_ax = plt.axes([0.3985294117647059, 0.225, 0.2279411764705882, 0.77])
cb_dm = plt.axes([0.64, 0.225, 0.01, 0.2])
mean_ax = plt.axes([0.75, 0.6, 0.23, 0.25])
cov_ax = plt.axes([0.75, 0.1, 0.23, 0.25])

subj_ax.plot((0,25*24), (0, 1e-12*24), color=[0.8, 0.8, 0.8])
for ii in range(24):
    d = data.data[I[ii],0, :, 23]
    subj_ax.plot(np.arange(96)+25*ii, d + 1e-12*ii)
for tag in ['top','right']:
    subj_ax.spines[tag].set_visible(False)
subj_ax.spines['bottom'].set_bounds(0, 96)
subj_ax.spines['left'].set_bounds(0, 0.5e-11)
subj_ax.set_xlim(0)
subj_ax.set_ylim(0)
subj_ax.set_xticks([0, 48, 96], [0, 24, 48])
subj_ax.set_yticks([0, 0.5e-11])
l = subj_ax.set_xlabel('Frequency (Hz)')
l.set_position((0.1, 44.76166666666667))
l = subj_ax.set_ylabel('Power')
l.set_position((41.91666666666667, 0.05))
subj_ax.text(48+25*16, 1e-12*12, r'Participants $\rightarrow$', rotation=55)
qlt.subpanel_label(subj_ax, chr(65), yf=0.75)


pcm = plot_design(des_ax, design.design_matrix[I[:24], :], design.regressor_names)
des_ax.set_ylim(0, 37.5)
des_ax.spines['left'].set_bounds(0, 24)
des_ax.set_yticks(np.arange(5)*5)
des_ax.set_xticklabels(['Mean', 'Num Blinks', 'Sex', ''])
des_ax.set_ylabel('Participants')
qlt.subpanel_label(des_ax, chr(65+1), yf=0.75)
plt.colorbar(pcm, cax=cb_dm)


mean_ax.errorbar(freq_vect, gmodel.copes[0, 0, :, 23], yerr=np.sqrt(gmodel.varcopes[0, 0, :, 23]), errorevery=2)
mean_ax.set_title('Group Mean Spectrum')
qlt.decorate_spectrum(mean_ax)
qlt.subpanel_label(mean_ax, chr(65+2), yf=1.1)

cov_ax.errorbar(freq_vect, gmodel.copes[1, 0, :, 23], yerr=np.sqrt(gmodel.varcopes[1, 0, :, 23]), errorevery=2)
cov_ax.set_title('Group effects')
cov_ax.errorbar(freq_vect, gmodel.copes[2, 4, :, 23], yerr=np.sqrt(gmodel.varcopes[1, 4, :, 23]), errorevery=2)
cov_ax.legend(['Blinks', 'Sex'])
qlt.decorate_spectrum(cov_ax)
qlt.subpanel_label(cov_ax, chr(65+3), yf=1.1)


fout = os.path.join(outdir, 'lemon-group_glm-overview.png')
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

fout = os.path.join(outdir, 'lemon-group_glm-clusters.png')
plt.savefig(fout, transparent=True, dpi=300)

#%% ----------------------------
# Sanity check figure

plt.figure()
for ii in range(3):
    for jj in range(5):
        ind = (jj+1)+ii*5
        print(ind)
        plt.subplot(3, 5, ind)
        plt.plot(freq_vect, gmodel.copes[ii, jj, :, :])

#%% ----------------------------


