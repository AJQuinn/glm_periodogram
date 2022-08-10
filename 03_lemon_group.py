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
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append('/Users/andrew/src/qlt')
import qlt

from glm_config import cfg

#%% ---------------------------------------------------
# Load single subject for reference
from lemon_support import (lemon_set_channel_montage, lemon_ica, lemon_create_heog)

config = osl.preprocessing.load_config('lemon_preproc.yml')

# Drop preproc after montage - only really need the channel info
config['preproc'] = config['preproc'][:3]

base = '/Users/andrew/Projects/lemon/EEG_Raw_BIDS_ID/sub-010060/RSEEG'
infile = os.path.join(base, 'sub-010060.vhdr')
extras = [lemon_set_channel_montage, lemon_ica, lemon_create_heog]
dataset = osl.preprocessing.run_proc_chain(infile, config, extra_funcs=extras)

raw = dataset['raw'].pick_types(eeg=True)

#%% --------------------------------------------------
# Load first level results and fit group model

#inputs = os.path.join(cfg['lemon_analysis_dir'], 'lemon_eeg_sensorglm_groupdata.hdf5')
inputs = os.path.join('/Users/andrew/Projects/glm/glm_psd/analysis', 'lemon_eeg_sensorglm_groupdata.hdf5')
#inputs2 = os.path.join('/Users/andrew/Projects/glm/glm_psd/analysis', 'lemon_eeg_sensorglm_groupdata_conf.hdf5')
nulls = os.path.join('/Users/andrew/Projects/glm/glm_psd/analysis', 'lemon_eeg_sensorglm_groupdata_null.hdf5')

data = obj_from_hdf5file(inputs, 'data')
datanull = obj_from_hdf5file(nulls, 'data')
with h5py.File(inputs, 'r') as F:
    #freq_vect = F['freq_vect'][()]  # needs fixing server-side
    freq_vect = np.linspace(0.5, 100, 200)

# Drop obvious outliers
bads = sails.utils.detect_artefacts(data.data[:, 0, :, :], axis=0)
clean_data = data.drop(np.where(bads)[0])

# Load age and sex data
#df = pd.read_csv('/Users/andrew/Projects/ntad/RA_Interview/LEMON_RA_InterviewData2.csv')
meta_file = os.path.join(cfg['lemon_raw'], 'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
df = pd.read_csv(meta_file)
age = []
sex = []
for idx, subj in enumerate(data.info['subj_id']):
    subj = 'sub-' + subj
    ind = np.where(df['ID'] == subj)[0][0]
    row = df.iloc[ind]
    #age.append(row['Age'] + 2.5)
    age.append(float(row['Age'].split('-')[0]) + 2.5)
    sex.append(row['Gender_ 1=female_2=male'])

data.info['age'] = age
data.info['sex'] = sex
data.info['subj_id'] = list(data.info['subj_id'])

datanull.info['age'] = age
datanull.info['sex'] = sex
datanull.info['subj_id'] = list(datanull.info['subj_id'])

fl_regressor_names = ['Open', 'Closed', 'Linear', 'Bads', 'BadDiffs', 'V-EOG', 'H-EOG']
confs = h5py.File(inputs, 'r')['confs'][()]
for ii in range(5):
    data.info[fl_regressor_names[ii+2]] = confs[:, ii+2]
    datanull.info[fl_regressor_names[ii+2]] = confs[:, ii+2]

keeps = np.where(np.array(age) < 45)[0]
drops = np.setdiff1d(np.arange(len(age)), keeps)

data = data.drop(drops)
datanull = datanull.drop(drops)

# Refit group model
DC = glm.design.DesignConfig()
DC.add_regressor(name='Female', rtype='Categorical', datainfo='sex', codes=1)
DC.add_regressor(name='Male', rtype='Categorical', datainfo='sex', codes=2)
#DC.add_regressor(name='Blinks', rtype='Parametric', datainfo='num_blinks', preproc='z')
#for ii in range(5):
#    DC.add_regressor(name=fl_regressor_names[ii+2], rtype='Parametric', datainfo=fl_regressor_names[ii+2], preproc='z')
DC.add_contrast(name='GroupMean',values={'Female': 0.5, 'Male': 0.5})
DC.add_contrast(name='Female>Male',values={'Female': 1, 'Male': -1})
DC.add_simple_contrasts()

design = DC.design_from_datainfo(data.info)
gmodel = glm.fit.OLSModel(design, data)
gmodel_null = glm.fit.OLSModel(design, datanull)

# Housekeeping and rescaling
#fl_contrast_names = ['Mean', 'Linear Trend', 'Eyes Open>Closed', 'Bad Segments', 'VEOG', 'HEOG']
fl_contrast_names = ['Mean', 'Eyes Open>Closed', 'Eyes Open', 'Eyes Closed',
                     'Linear Trend', 'Bad Segments', 'Bad Segments Diff', 'VEOG', 'HEOG']
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

#def varcope_corr_medfilt(vc, window_size=11, smooth_dims=None):
#    if smooth_dims is None:
#        smooth_dims = np.arange(vc.ndim)
#    elif isinstance(smooth_dims, (float, int)):
#        smooth_dims = [smooth_dims]
#    print('Applying medfilt smoothing of {} to dims {} of {}'.format(window_size, smooth_dims, vc.shape))
#
#    sigma = np.ones((vc.ndim,), dtype=int)
#    sigma[np.array(smooth_dims)] = window_size
#
#    #return ndimage.median_filter(vc, size=sigma)
#    return ndimage.uniform_filter(vc, size=sigma)
#
#
#tstat_args = {'hat_factor': None, 'varcope_smoothing': 'medfilt',
#              'window_size': 15, 'smooth_dims': 2}
#ts = gmodel.get_tstats(**tstat_args)
#
#vc = varcope_corr_medfilt(gmodel.varcopes, window_size=15, smooth_dims=2)
#
#plt.figure()
#for ii in range(gmodel.num_contrasts):
#    plt.subplot(2,5,ii+1)
#    #plt.plot(gmodel.copes[ii, 0 ,:, :])
#    #plt.plot(gmodel.varcopes[ii, 0 ,:, :].mean(axis=1))
#    #plt.plot(vc[ii, 0 ,:, :].mean(axis=1))
#    plt.plot(ts[ii, 0 ,:, :])
#    plt.title(gmodel.contrast_names[ii])




#%% ------------------------------------------------------
# Effect of covariates

sw = h5py.File(inputs, 'r')['sw'][keeps]
sw_null = h5py.File(nulls, 'r')['sw'][keeps]
r2 = h5py.File(inputs, 'r')['r2'][keeps]
r2_null = h5py.File(nulls, 'r')['r2'][keeps]

labs = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
fontargs = {'fontsize': 'large', 'fontweight': 'bold', 'color':'r', 'ha':'center', 'va': 'center'}

plt.figure(figsize=(16, 9))

ppts = [22, 66, 5, 6, 11]
plt.axes((0.1, 0.5, 0.4*(9/16), 0.4))
plt.plot(r2_null, r2, 'ko')
plt.plot(r2_null[ppts], r2[ppts], 'ro')
plt.text(r2_null[ppts][0]-0.01, r2[ppts][0]+0.02, labs[0], **fontargs)
plt.text(r2_null[ppts][1]-0.02, r2[ppts][1]+0.02, labs[1], **fontargs)
plt.text(r2_null[ppts][2]+0.02, r2[ppts][2]-0.035, labs[2],      **fontargs)
plt.text(r2_null[ppts][3]+0.035, r2[ppts][3], labs[3],      **fontargs)
plt.text(r2_null[ppts][4]-0.02, r2[ppts][4]+0.02, labs[4], **fontargs)

plt.plot((0, 1), (0, 1), 'k')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.xlabel('Null Model')
plt.ylabel('Full Model')
plt.xlim(0.2, 0.8)
plt.ylim(0.2, 0.8)
qlt.subpanel_label(plt.gca(), 'A')
plt.title('R-Squared\nhigh values indicate greater variance explained')

plt.axes((0.375, 0.5, 0.05, 0.4))
plt.boxplot(r2-r2_null)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.ylim(-0.5, 0.5)
plt.xticks([1], ['Full Model minus \nNull Model'])

plt.axes((0.6, 0.5, 0.4*(9/16), 0.4))
plt.text(sw_null[ppts][0]-0.02, sw[ppts][0]+0.02, labs[0], **fontargs)
plt.text(sw_null[ppts][1]-0.02, sw[ppts][1]+0.02, labs[1], **fontargs)
plt.text(sw_null[ppts][2], sw[ppts][2]-0.035, labs[2],      **fontargs)
plt.text(sw_null[ppts][3]+0.035, sw[ppts][3], labs[3],      **fontargs)
plt.text(sw_null[ppts][4]-0.02, sw[ppts][4]+0.02, labs[4], **fontargs)
plt.plot(sw_null, sw, 'ko')
plt.plot(sw_null[ppts], sw[ppts], 'ro')

plt.plot((0, 1), (0, 1), 'k')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.xlabel('Null Model')
plt.ylabel('Full Model')
plt.xlim(0.2, 1)
plt.ylim(0.2, 1)
qlt.subpanel_label(plt.gca(), 'B')
plt.title('Shapiro-Wilks\nvalues closer to one indicate more gaussian residuals')

plt.axes((0.875, 0.5, 0.05, 0.4))
plt.boxplot(sw-sw_null)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.ylim(-0.2, 0.2)
plt.xticks([1], ['Full Model minus \nNull Model'])

fx = qlt.prep_scaled_freq(0.5, freq_vect,)
labs = ['C i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
for ii in range(len(ppts[:5])):
    ax = plt.axes((0.15+ii*0.16, 0.05, 0.1, 0.3))
    ax.plot(fx[0], datanull.data[ppts[ii], 0, : :].mean(axis=1))
    ax.plot(fx[0], data.data[ppts[ii], 0, : :].mean(axis=1))
    ax.set_xticks(fx[2][::2])
    ax.set_xticklabels(fx[1][::2])
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.ylim(0, 1.2e-5)
    qlt.subpanel_label(plt.gca(), labs[ii])
    if ii == 0:
        plt.ylabel('FFT Magnitude')
    plt.xlabel('Frequency (Hz)')
plt.legend(['Null Model', 'Full Model'], frameon=False)

fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_stats-summary.png')
plt.savefig(fout, transparent=True, dpi=300)

#%% ------------------------------------------------------
# Permutation stats - run or load from disk

adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(raw.info, 'eeg')
ntests = np.prod(data.data.shape[2:])
ntimes = data.data.shape[2]
adjacency = mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)

cft = 3
#cft = -stats.t.ppf(0.01, data.num_observations)
tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt',
              'window_size': 15, 'smooth_dims': 1}
#tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt',
#              'window_size': 15, 'smooth_dims': 1}

# Permute
# Blinks on Mean, Mean on linear, task, blinks, bads
P = []

to_permute = [(0, 0, 'Overall Mean'),
              (0, 1, 'Group Mean of Open>Closed'),
              (0, 2, 'Group Mean of Open'),
              (0, 3, 'Group Mean of Closed'),
              (0, 4, 'Group Mean of Linear Trend'),
              (0, 5, 'Group Mean of Bad Segments'),
              (0, 6, 'Group Mean of Bad Segments Diff'),
              (0, 7, 'Group Mean of VEOG'),
              (0, 8, 'Group Mean of HEOG'),
              (1, 0, 'Group Effect of Sex on Mean'),
              (1, 1, 'Group Effect of Sex on Open>Closed'),
              (1, 2, 'Group Effect of Sex on Open'),
              (1, 3, 'Group Effect of Sex on Closed')]


run_perms = False
standardise = False
for icon in range(len(to_permute)):
    if run_perms:
        gl_con = to_permute[icon][0]
        fl_con = to_permute[icon][1]
        # Only working with mean regressor for the moment
        fl_mean_data = deepcopy(data)
        fl_mean_data.data = data.data[:, fl_con, : ,:]

        if standardise:
            #mn = fl_data.data.mean(axis=1)[:, None, :]
            #st = fl_data.data.std(axis=1)[:, None, :]
            factor = fl_mean_data.data.sum(axis=1)[:, None, :]
            fl_mean_data.data = fl_mean_data.data / factor

        p = glm.permutations.MNEClusterPermutation(design, fl_mean_data, gl_con, 150,
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

# Permuate
# Blinks on Mean, Mean on linear, task, blinks, bads
Pn = []

run_perms = False
for icon in range(len(to_permute)):
    if to_permute[icon][1] > 3:
        Pn.append(None)
    elif run_perms:
        gl_con = to_permute[icon][0]
        fl_con = to_permute[icon][1]
        # Only working with mean regressor for the moment
        fl_mean_data = deepcopy(datanull)
        fl_mean_data.data = datanull.data[:, fl_con, : ,:]

        p = glm.permutations.MNEClusterPermutation(design, fl_mean_data, gl_con, 100,
                                                   nprocesses=4,
                                                   metric='tstats',
                                                   cluster_forming_threshold=cft,
                                                   tstat_args=tstat_args,
                                                   adjacency=adjacency)

        with open(os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_permsnull-con{0}.pkl'.format(icon)), "wb") as dill_file:
            dill.dump(p, dill_file)

        Pn.append(p)
    else:
        dill_file = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_permsnull-con{0}.pkl'.format(icon))
        Pn.append(dill.load(open(dill_file, 'rb')))

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

#subj_ax = plt.axes([0.05, 0.125, 0.35, 0.35*aspect])
subj_ax = plt.axes([0.05, 0.125, 0.35, 0.75])
des_ax = plt.axes([0.425, 0.225, 0.175, 0.51])
cb_dm = plt.axes([0.62, 0.225, 0.01, 0.2])
mean_ax = plt.axes([0.75, 0.6, 0.23, 0.25])
cov_ax = plt.axes([0.75, 0.1, 0.23, 0.25])

xstep = 35
ystep = 2e-6
ntotal = 36
subj_ax.plot((0, xstep*ntotal), (0, ystep*ntotal), color=[0.8, 0.8, 0.8], lw=0.5)
for ii in range(28):
    d = data.data[I[ii],0, :, 23]
    ii = ii + 8 if ii > 14 else ii
    subj_ax.plot(np.arange(len(freq_vect))+xstep*ii, d + ystep*ii)
for tag in ['top','right']:
    subj_ax.spines[tag].set_visible(False)
subj_ax.spines['bottom'].set_bounds(0, len(freq_vect))
subj_ax.spines['left'].set_bounds(0, 1e-5)
subj_ax.set_xlim(0)
subj_ax.set_ylim(0)
subj_ax.set_xticks([])
subj_ax.set_yticks([])
l = subj_ax.set_xlabel(r'Frequency (Hz) $\rightarrow$', loc='left')
l = subj_ax.set_ylabel(r'Amplitude $\rightarrow$', loc='bottom')
subj_ax.text(48+35*18, ystep*19, '...', fontsize='xx-large', rotation=52)
subj_ax.text(48+35*18, ystep*16, r'Participants $\rightarrow$', rotation=52)
qlt.subpanel_label(subj_ax, chr(65), yf=0.75, xf=0.05)
subj_ax.text(0.125, 0.725, 'First Level GLM Spectra', transform=subj_ax.transAxes, fontsize='large')

pcm = plot_design(des_ax, design.design_matrix[:, :], design.regressor_names)
des_ax.set_xticklabels(['Female', 'Male', ''])
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
cov_ax.set_title('Sex effect on Mean Spectrum')
cov_ax.set_xticks(xf[2], xf[1])
qlt.decorate_spectrum(cov_ax, ylabel='Amplitude')
qlt.subpanel_label(cov_ax, chr(65+3), yf=1.1)

fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_glm-overview.png')
plt.savefig(fout, transparent=True, dpi=300)


#%% ----------------------------
fx = qlt.prep_scaled_freq(0.5, freq_vect,)

Q = P

plt.figure(figsize=(16, 12))

ax = plt.axes((0.075, 0.675, 0.3, 0.25))
qlt.plot_joint_spectrum(ax, gmodel.copes[0, 2, :, :], raw, freq_vect, base=0.5, freqs=[10, 22],
                        topo_scale=None, title='GroupMean Eyes Open', ylabel='FFT Magnitude')
qlt.subpanel_label(ax, 'A')

ax = plt.axes((0.45, 0.675, 0.3, 0.25))
qlt.plot_joint_spectrum(ax, gmodel.copes[0, 3, :, :], raw, freq_vect, base=0.5, freqs=[10, 22],
                        topo_scale=None, title='GroupMean Eyes Closed', ylabel='FFT Magnitude')
qlt.subpanel_label(ax, 'B')

ax = plt.axes([0.775, 0.7, 0.15, 0.15])
qlt.plot_channel_layout(ax, raw, size=100)

ll = [['Rec Start', 'Rec End'],
      ['Good Seg', 'Bad Seg'],
      ['Good Seg', 'Bad Seg'],
      ['Low V-EOG Activity', 'High V-EOG Activity'],
      ['Low H-EOG Activity', 'High H-EOG Activity']]

plt.subplots_adjust(left=0.05, right=0.95, wspace=0.45, hspace=0.5)

for ii in range(5):
    ax = plt.subplot(4,5,ii+11)

    fl_mean_data = deepcopy(data)
    fl_mean_data.data = data.data[:, to_permute[ii+4][1], : ,:]  # Mean

    qlt.plot_sensorspace_clusters(fl_mean_data, P[ii+4], raw, ax, xvect=freq_vect, base=0.5)
    ax.set_ylabel('t-stat')
    qlt.subpanel_label(ax, chr(65+ii+2))
    plt.title('Group Mean of {0}'.format(fl_contrast_names[ii+4]))

    ax = plt.subplot(4,5,ii+16)

    beta0 = gmodel.betas[:, 2:4, :, :].mean(axis=(0,1))
    beta1 = gmodel.betas[:, ii+4, :, :].mean(axis=0)

    if ii == 0:
        ax.plot(fx[0], np.mean(beta0 + -1*beta1, axis=1))
    else:
        ax.plot(fx[0], np.mean(beta0 + 0*beta1, axis=1))
    ax.plot(fx[0], np.mean(beta0 + 1*beta1, axis=1))
    ax.set_xticks(fx[2], fx[1])
    qlt.decorate_spectrum(ax)

    plt.legend(ll[ii], frameon=False)


fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_group-glm-meancov.png')
plt.savefig(fout, transparent=True, dpi=300)


#%% ----------------------------

plt.figure(figsize=(16, 9))

ax = plt.axes([0.075, 0.5, 0.175, 0.4])
fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 1, : ,:]  # Open>Closed
qlt.plot_sensorspace_clusters(fl_mean_data, P[1], raw, ax, xvect=freq_vect, base=0.5)
ax.set_ylabel('t-stat')
qlt.subpanel_label(ax, 'A')
plt.title('Within Subject Effect\n{0}_{1}'.format(gmodel.contrast_names[0], fl_contrast_names[1]))

# Open mean & Closed Mean
ax = plt.axes([0.075, 0.1, 0.175, 0.30])
plt.plot(fx[0], gmodel.copes[0, 2, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[0, 3, : ,:].mean(axis=1))
qlt.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
plt.legend(['Eyes Open', 'Eyes Closed'], frameon=False)

ax = plt.axes([0.3125, 0.5, 0.175, 0.4])
fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 0, : ,:]  # Mean
qlt.plot_sensorspace_clusters(fl_mean_data, P[9], raw, ax, xvect=freq_vect, base=0.5)
ax.set_ylabel('t-stat')
qlt.subpanel_label(ax, 'B')
plt.title('Between Subject Effect\n{0}_{1}'.format(gmodel.contrast_names[1], fl_contrast_names[0]))

# Female Mean & Male Mean
ax = plt.axes([0.3125, 0.1, 0.175, 0.3])
plt.plot(fx[0], gmodel.copes[2, 0, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 0, : ,:].mean(axis=1))
qlt.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
plt.legend(['Female', 'Male'], frameon=False)

ax = plt.axes([0.55, 0.5, 0.175, 0.4])
fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 1, : ,:]  # Mean
qlt.plot_sensorspace_clusters(fl_mean_data, P[10], raw, ax, xvect=freq_vect, base=0.5)
ax.set_ylabel('t-stat')
qlt.subpanel_label(ax, 'C')
plt.title('Interaction Effect\n{0}_{1}'.format(gmodel.contrast_names[1], fl_contrast_names[1]))

# Female Mean & Male Mean
ax = plt.axes([0.55, 0.1, 0.175, 0.3])
plt.plot(fx[0], gmodel.copes[2, 1, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 1, : ,:].mean(axis=1))
qlt.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
plt.legend(['Female : Open > Closed', 'Male : Open > Closed'], frameon=False)

ax = plt.axes([0.775, 0.7, 0.2, 0.2])
qlt.plot_channel_layout(ax, raw, size=100)

#ax = plt.subplot(4,4,(12,16))
ax = plt.axes([0.8, 0.175, 0.175, 0.35])
plt.plot(fx[0], gmodel.copes[2, 2, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 2, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[2, 3, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 3, : ,:].mean(axis=1))
plt.legend(['{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[3]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[3])], frameon=False)
qlt.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
qlt.subpanel_label(ax, 'D')

fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_group-glm-anova.png')
plt.savefig(fout, transparent=True, dpi=300)

#%% ----------------------------

import emd
def get_peaks(f, pxx, frange=None):

    if frange is not None:
        inds = np.logical_and((f > frange[0]), (f < frange[1]))
    else:
        inds = np.ones_like(pxx).astype(bool)

    x = pxx[inds]

    # Find peaks
    l, p = emd._sift_core._find_extrema(x, parabolic_extrema=True)

    if len(l) == 0:
        return np.nan, np.nan

    ind = np.argmax(p)
    p = p[ind]
    l = l[ind]

    fpara = np.diff(f)[0] * (l + np.where(inds)[0][0])

    return fpara, p

blue_dict =  {'patch_artist': True,
              'boxprops': dict(color='b', facecolor='b'),
              'capprops': dict(color='b'),
              'flierprops': dict(color='b', markeredgecolor='b'),
              'medianprops': dict(color='k'),
              'whiskerprops': dict(color='b')}
red_dict =  {'patch_artist': True,
             'boxprops': dict(color='r', facecolor='r'),
             'capprops': dict(color='r'),
             'flierprops': dict(color='r', markeredgecolor='r'),
             'medianprops': dict(color='k'),
             'whiskerprops': dict(color='r')}

plt.figure(figsize=(9, 9))

ff = np.zeros((2, 113))
for ii in range(113):
    ff[:, ii] = get_peaks(freq_vect, data.data[ii, 2, :, :].mean(axis=1), frange=(7,15))

plt.axes([0.1, 0.35, 0.35, 0.55])
plt.plot(ff[0, data.info['sex']==1], ff[1, data.info['sex']==1], 'or', alpha=1/3, mec='white')
plt.plot(ff[0, data.info['sex']==2], ff[1, data.info['sex']==2], 'ob', alpha=1/3, mec='white')
plt.plot(freq_vect, data.data[data.info['sex']==1, 2, :, :].mean(axis=(0, 2)), 'r', lw=3)
plt.plot(freq_vect, data.data[data.info['sex']==2, 2, :, :].mean(axis=(0, 2)), 'b', lw=3)
plt.xlim(0, 20)
plt.ylim(0, 1.2e-5)
l = plt.legend(['Female Subject Peak', 'Male Subject Peak', 'Female Group Spectrum', 'Male Group Spectrum'], bbox_to_anchor=(1, 1))
qlt.decorate_spectrum(plt.gca(), ylabel='FFT Magnitude')
plt.title('Eyes Open Rest')
qlt.subpanel_label(plt.gca(), 'A')
plt.xticks(np.linspace(0,20,5))

plt.axes([0.1, 0.075, 0.35, 0.2])
yy = ff[0, data.info['sex']==2]
yy = yy[numpy.logical_not(np.isnan(yy))]
plt.boxplot(yy, vert=False, positions=[0.8], **blue_dict)
xx = ff[0, data.info['sex']==1]
xx = xx[numpy.logical_not(np.isnan(xx))]
plt.boxplot(xx, vert=False, positions=[1], **red_dict)
plt.xlim(6, 14)
plt.ylim(0.6, 1.2)
plt.gca().set_yticklabels(('Male', 'Female'))
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.gca().spines['left'].set_bounds(0.7, 1.1)
plt.xlabel('Frequency (Hz)')
qlt.subpanel_label(plt.gca(), 'B')

tt = stats.ttest_ind(xx, yy)
print('Eyes Open')
print('Male mean : {}'.format(yy.mean()))
print('Female mean : {}'.format(xx.mean()))
print('t({0})={1}, p={2}'.format(len(xx)+len(yy)-2, tt.statistic, tt.pvalue))

ff = np.zeros((2, 113))
for ii in range(113):
    ff[:, ii] = get_peaks(freq_vect, data.data[ii, 3, :, :].mean(axis=1), frange=(7,15))

plt.axes([0.55, 0.35, 0.35, 0.55])
plt.plot(ff[0, data.info['sex']==1], ff[1, data.info['sex']==1], 'or', alpha=1/3, mec='white')
plt.plot(ff[0, data.info['sex']==2], ff[1, data.info['sex']==2], 'ob', alpha=1/3, mec='white')
plt.plot(freq_vect, data.data[data.info['sex']==1, 3, :, :].mean(axis=(0, 2)), 'r', lw=3)
plt.plot(freq_vect, data.data[data.info['sex']==2, 3, :, :].mean(axis=(0, 2)), 'b', lw=3)
plt.xlim(0, 20)
plt.ylim(0, 1.2e-5)
qlt.decorate_spectrum(plt.gca(), ylabel='FFT Magnitude')
plt.title('Eyes Closed Rest')
qlt.subpanel_label(plt.gca(), 'C')
plt.xticks(np.linspace(0,20,5))

plt.axes([0.55, 0.075, 0.35, 0.2])
yy = ff[0, data.info['sex']==2]
yy = yy[numpy.logical_not(np.isnan(yy))]
plt.boxplot(yy, vert=False, positions=[0.8], **blue_dict)
xx = ff[0, data.info['sex']==1]
xx = xx[numpy.logical_not(np.isnan(xx))]
plt.boxplot(xx, vert=False, positions=[1], **red_dict)
plt.xlim(6, 14)
plt.ylim(0.6, 1.2)
plt.gca().set_yticklabels(('Male', 'Female'))
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.gca().spines['left'].set_bounds(0.7, 1.1)
plt.xlabel('Frequency (Hz)')
qlt.subpanel_label(plt.gca(), 'D')

tt = stats.ttest_ind(xx, yy)
print('Eyes Closed')
print('Male mean : {}'.format(yy.mean()))
print('Female mean : {}'.format(xx.mean()))
print('t({0})={1}, p={2}'.format(len(xx)+len(yy)-2, tt.statistic, tt.pvalue))

fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_group-sex-peaks.png')
plt.savefig(fout, transparent=True, dpi=300)

eye

#%% ----------------------------
fx = qlt.prep_scaled_freq(0.5, freq_vect,)

Q = P

plt.figure(figsize=(16, 12))

ax = plt.axes((0.15, 0.65, 0.3, 0.3))
qlt.plot_joint_spectrum(ax, gmodel.copes[0, 2, :, :], raw, freq_vect, base=0.5, freqs=[10, 22],
                        topo_scale=None, title='GroupMean Eyes Open', ylabel='FFT Magnitude')
qlt.subpanel_label(ax, 'A')

ax = plt.axes((0.55, 0.65, 0.3, 0.3))
qlt.plot_joint_spectrum(ax, gmodel.copes[0, 3, :, :], raw, freq_vect, base=0.5, freqs=[10, 22],
                        topo_scale=None, title='GroupMean Eyes Closed', ylabel='FFT Magnitude')
qlt.subpanel_label(ax, 'B')

ax = plt.axes([0.075, 0.3, 0.175, 0.25])
fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 1, : ,:]  # Open>Closed
qlt.plot_sensorspace_clusters(fl_mean_data, P[1], raw, ax, xvect=freq_vect, base=0.5)
ax.set_ylabel('t-stat')
qlt.subpanel_label(ax, 'C')
plt.title('Within Subject Effect\n{0}_{1}'.format(gmodel.contrast_names[0], fl_contrast_names[1]))

# Open mean & Closed Mean
ax = plt.axes([0.075, 0.1, 0.175, 0.15])
plt.plot(fx[0], gmodel.copes[0, 2, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[0, 3, : ,:].mean(axis=1))
qlt.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
plt.legend(['Eyes Open', 'Eyes Closed'], frameon=False)

ax = plt.axes([0.3125, 0.3, 0.175, 0.25])
fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 0, : ,:]  # Mean
qlt.plot_sensorspace_clusters(fl_mean_data, P[9], raw, ax, xvect=freq_vect, base=0.5)
ax.set_ylabel('t-stat')
qlt.subpanel_label(ax, 'D')
plt.title('Between Subject Effect\n{0}_{1}'.format(gmodel.contrast_names[1], fl_contrast_names[0]))

# Female Mean & Male Mean
ax = plt.axes([0.3125, 0.1, 0.175, 0.15])
plt.plot(fx[0], gmodel.copes[2, 0, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 0, : ,:].mean(axis=1))
qlt.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
plt.legend(['Female', 'Male'], frameon=False)

ax = plt.axes([0.55, 0.3, 0.175, 0.25])
fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 1, : ,:]  # Mean
qlt.plot_sensorspace_clusters(fl_mean_data, P[10], raw, ax, xvect=freq_vect, base=0.5)
ax.set_ylabel('t-stat')
qlt.subpanel_label(ax, 'E')
plt.title('Interaction Effect\n{0}_{1}'.format(gmodel.contrast_names[1], fl_contrast_names[1]))

# Female Mean & Male Mean
ax = plt.axes([0.55, 0.1, 0.175, 0.15])
plt.plot(fx[0], gmodel.copes[2, 1, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 1, : ,:].mean(axis=1))
qlt.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
plt.legend(['Female : Open > Closed', 'Male : Open > Closed'], frameon=False)

#ax = plt.subplot(4,4,(12,16))
ax = plt.axes([0.8, 0.175, 0.175, 0.25])
plt.plot(fx[0], gmodel.copes[2, 2, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 2, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[2, 3, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 3, : ,:].mean(axis=1))
plt.legend(['{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[3]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[3])], frameon=False)
qlt.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
qlt.subpanel_label(ax, 'F')


fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_group-glm-sex.png')
plt.savefig(fout, transparent=True, dpi=300)


#%% ----------------------------

ll = [['Rec Start', 'Rec End'],
      ['Good Seg', 'Bad Seg'],
      ['Good Seg', 'Bad Seg'],
      ['Low V-EOG Activity', 'High V-EOG Activity'],
      ['Low H-EOG Activity', 'High H-EOG Activity']]


plt.figure(figsize=(16, 9))
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.45, hspace=0.5)

for ii in range(5):
    ax = plt.subplot(3,5,ii+6)

    fl_mean_data = deepcopy(data)
    fl_mean_data.data = data.data[:, to_permute[ii+4][1], : ,:]  # Mean

    qlt.plot_sensorspace_clusters(fl_mean_data, P[ii+4], raw, ax, xvect=freq_vect, base=0.5)
    ax.set_ylabel('t-stat')
    qlt.subpanel_label(ax, chr(65+ii))
    plt.title('Group Mean of {0}'.format(fl_contrast_names[ii+4]))

    ax = plt.subplot(3,5,ii+11)

    beta0 = gmodel.betas[:, 2:4, :, :].mean(axis=(0,1))
    beta1 = gmodel.betas[:, ii+4, :, :].mean(axis=0)

    if ii == 0:
        ax.plot(fx[0], np.mean(beta0 + -1*beta1, axis=1))
    else:
        ax.plot(fx[0], np.mean(beta0 + 0*beta1, axis=1))
    ax.plot(fx[0], np.mean(beta0 + 1*beta1, axis=1))
    ax.set_xticks(fx[2], fx[1])
    qlt.decorate_spectrum(ax)

    plt.legend(ll[ii], frameon=False)

fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_group-glm-covs.png')
plt.savefig(fout, transparent=True, dpi=300)

#%% ----------------------------
fx = qlt.prep_scaled_freq(0.5, freq_vect,)

Q = P

plt.figure(figsize=(16, 9))

ax = plt.axes((0.1, 0.55, 0.15, 0.35))
qlt.plot_joint_spectrum(ax, gmodel.copes[0, 2, :, :], raw, freq_vect, base=0.5, freqs=[10, 22],
                        topo_scale=None, title='Group Average Eyes Open', ylabel='FFT Magnitude')

ax = plt.axes((0.275, 0.55, 0.15, 0.35))
pcolormesh(fx[0], np.arange(113), np.log(data.data[:, 2, :, :].mean(axis=2)), cmap='cividis', vmax=-10)
plt.xticks(fx[2], fx[1])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Subjects')

ax = plt.axes((0.55, 0.55, 0.15, 0.35))
qlt.plot_joint_spectrum(ax, gmodel.copes[0, 3, :, :], raw, freq_vect, base=0.5, freqs=[10, 22],
                        topo_scale=None, title='Group Average Eyes Closed', ylabel='FFT Magnitude')

ax = plt.axes((0.725, 0.55, 0.15, 0.35))
pcolormesh(fx[0], np.arange(113), np.log(data.data[:, 3, :, :].mean(axis=2)), cmap='cividis', vmax=-10)
plt.xticks(fx[2], fx[1])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Subjects')

ax = plt.axes((0.05, 0.05, 0.2, 0.35))
pind = 1
gl, fl, name = to_permute[pind]
fl_data = deepcopy(data)
fl_data.data = fl_data.data[:, fl, :, :]
qlt.plot_sensorspace_clusters(fl_data, Q[pind], raw, ax, xvect=freq_vect, base=0.5, title=name)
print(name)
print('{} : {} - {} : {}'.format(fl, fl_contrast_names[fl],  Q[pind].contrast_idx, gmodel.contrast_names[Q[pind].contrast_idx]))

ax = plt.axes((0.3, 0.05, 0.2, 0.35))
pind = 10
gl, fl, name = to_permute[pind]
fl_data = deepcopy(data)
fl_data.data = fl_data.data[:, fl, :, :]
qlt.plot_sensorspace_clusters(fl_data, Q[pind], raw, ax, xvect=freq_vect, base=0.5, title=name)
print(name)
print('{} : {} - {} : {}'.format(fl, fl_contrast_names[fl],  Q[pind].contrast_idx, gmodel.contrast_names[Q[pind].contrast_idx]))

ax = plt.axes((0.525, 0.05, 0.2, 0.35))
pind = 11
gl, fl, name = to_permute[pind]
fl_data = deepcopy(data)
fl_data.data = fl_data.data[:, fl, :, :]
qlt.plot_sensorspace_clusters(fl_data, Q[pind], raw, ax, xvect=freq_vect, base=0.5, title=name)
print(name)
print('{} : {} - {} : {}'.format(fl, fl_contrast_names[fl],  Q[pind].contrast_idx, gmodel.contrast_names[Q[pind].contrast_idx]))

ax = plt.axes((0.775, 0.05, 0.2, 0.35))
pind = 12
gl, fl, name = to_permute[pind]
fl_data = deepcopy(data)
fl_data.data = fl_data.data[:, fl, :, :]
qlt.plot_sensorspace_clusters(fl_data, Q[pind], raw, ax, xvect=freq_vect, base=0.5, title=name)
print(name)
print('{} : {} - {} : {}'.format(fl, fl_contrast_names[fl],  Q[pind].contrast_idx, gmodel.contrast_names[Q[pind].contrast_idx]))

#%% ----------------------------

plt.figure(figsize=(16, 9))
ax = plt.subplot(2,4,1)
qlt.plot_sensor_spectrum(ax, gmodel.copes[2, 2, :, :], raw, freq_vect, base=0.5)
plt.title('{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[2]))
ax = plt.subplot(2,4,2)
qlt.plot_sensor_spectrum(ax, gmodel.copes[3, 2, :, :], raw, freq_vect, base=0.5)
plt.title('{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[2]))
ax = plt.subplot(2,4,5)
qlt.plot_sensor_spectrum(ax, gmodel.copes[2, 3, :, :], raw, freq_vect, base=0.5)
plt.title('{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[3]))
ax = plt.subplot(2,4,6)
qlt.plot_sensor_spectrum(ax, gmodel.copes[3, 3, :, :], raw, freq_vect, base=0.5)
plt.title('{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[3]))
ax = plt.subplot(2,4,3)
qlt.plot_sensor_spectrum(ax, gmodel.copes[0, 1, :, :], raw, freq_vect, base=0.5)
ax = plt.subplot(2,4,7)
qlt.plot_sensor_spectrum(ax, gmodel.copes[1, 0, :, :], raw, freq_vect, base=0.5)
ax = plt.subplot(2,4,4)
plt.plot(xf[0], gmodel.copes[2, 2, : ,:].mean(axis=1))
plt.plot(xf[0], gmodel.copes[3, 2, : ,:].mean(axis=1))
plt.plot(xf[0], gmodel.copes[2, 3, : ,:].mean(axis=1))
plt.plot(xf[0], gmodel.copes[3, 3, : ,:].mean(axis=1))
plt.legend(['{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[3]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[3])])
plt.xticks(xf[2], xf[1])
ax = plt.subplot(2,4,8)
qlt.plot_sensor_spectrum(ax, gmodel.copes[1, 1, :, :], raw, freq_vect, base=0.5)
plt.title('{0}_{1}'.format(gmodel.contrast_names[1], fl_contrast_names[1]))


plt.figure(figsize=(16, 9))
#ax = plt.subplot(121)
ax = plt.axes([0.075, 0.1, 0.35, 0.77])
plt.plot(xf[0], gmodel.copes[2, 2, : ,:].mean(axis=1))
plt.plot(xf[0], gmodel.copes[3, 2, : ,:].mean(axis=1))
plt.plot(xf[0], gmodel.copes[2, 3, : ,:].mean(axis=1))
plt.plot(xf[0], gmodel.copes[3, 3, : ,:].mean(axis=1))
plt.legend(['{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[3]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[3])], frameon=False, loc=3)
plt.xticks(xf[2], xf[1])
qlt.decorate_spectrum(ax)
axins = ax.inset_axes([0.65, 0.4, 0.45, 0.55])
axins.plot(xf[0], gmodel.copes[2, 2, : ,:].mean(axis=1))
axins.plot(xf[0], gmodel.copes[3, 2, : ,:].mean(axis=1))
axins.plot(xf[0], gmodel.copes[2, 3, : ,:].mean(axis=1))
axins.plot(xf[0], gmodel.copes[3, 3, : ,:].mean(axis=1))
axins.set_xticks(xf[2], xf[1])
axins.set_xlim(2, 4)
axins.set_ylim(1e-6, 4e-6)
qlt.decorate_spectrum(axins)
#axins.set_yticks(np.arange(4)*0.0001 + 0.0004)
axins.set_ylabel('')
axins.set_xlabel('')
ax.indicate_inset_zoom(axins, edgecolor="black")
qlt.subpanel_label(ax, 'A')
ax = plt.subplot(4,4,7)
#qlt.plot_sensor_spectrum(ax, gmodel.copes[0, 1, :, :], raw, freq_vect, base=0.5)
fl_mean_data = deepcopy(datanull)
fl_mean_data.data = datanull.data[:, 1, : ,:]  # Open>Closed
qlt.plot_sensorspace_clusters(fl_mean_data, P[1], raw, ax, xvect=freq_vect, base=0.5)
qlt.subpanel_label(ax, 'B')
plt.title('{0}_{1}'.format(gmodel.contrast_names[0], fl_contrast_names[1]))
ax = plt.subplot(4,4,15)
#qlt.plot_sensor_spectrum(ax, gmodel.copes[1, 0, :, :], raw, freq_vect, base=0.5)
fl_mean_data = deepcopy(datanull)
fl_mean_data.data = datanull.data[:, 0, : ,:]  # Mean
qlt.plot_sensorspace_clusters(fl_mean_data, P[9], raw, ax, xvect=freq_vect, base=0.5)
qlt.subpanel_label(ax, 'C')
plt.title('{0}_{1}'.format(gmodel.contrast_names[1], fl_contrast_names[0]))

ax = plt.subplot(2,4,8)
fl_mean_data = deepcopy(datanull)
fl_mean_data.data = datanull.data[:, 1, : ,:]  # Open>Closed
qlt.plot_sensorspace_clusters(fl_mean_data, P[10], raw, ax, xvect=freq_vect, base=0.5)
qlt.subpanel_label(ax, 'D')
plt.title('{0}_{1}'.format(gmodel.contrast_names[1], fl_contrast_names[1]))


plt.figure()

ax = plt.subplot(2,4,1)
#qlt.plot_sensor_spectrum(ax, gmodel.copes[0, 1, :, :], raw, freq_vect, base=0.5)
fl_mean_data = deepcopy(datanull)
fl_mean_data.data = datanull.data[:, 1, : ,:]  # Open>Closed
qlt.plot_sensorspace_clusters(fl_mean_data, P[1], raw, ax, xvect=freq_vect, base=0.5)
qlt.subpanel_label(ax, 'B')
plt.title('{0}_{1}'.format(gmodel.contrast_names[0], fl_contrast_names[1]))

# Open mean & Closed Mean
plt.subplot(2,4,5)
plt.plot(fx[0], gmodel.copes[0, 2, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[0, 3, : ,:].mean(axis=1))

ax = plt.subplot(2,4,2)
#qlt.plot_sensor_spectrum(ax, gmodel.copes[1, 0, :, :], raw, freq_vect, base=0.5)
fl_mean_data = deepcopy(datanull)
fl_mean_data.data = datanull.data[:, 0, : ,:]  # Mean
qlt.plot_sensorspace_clusters(fl_mean_data, P[9], raw, ax, xvect=freq_vect, base=0.5)
qlt.subpanel_label(ax, 'C')
plt.title('{0}_{1}'.format(gmodel.contrast_names[1], fl_contrast_names[0]))

# Female Mean & Male Mean
plt.subplot(2,4,6)
plt.plot(fx[0], gmodel.copes[2, 0, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 0, : ,:].mean(axis=1))

ax = plt.subplot(2,4,3)
#qlt.plot_sensor_spectrum(ax, gmodel.copes[1, 0, :, :], raw, freq_vect, base=0.5)
fl_mean_data = deepcopy(datanull)
fl_mean_data.data = datanull.data[:, 1, : ,:]  # Mean
qlt.plot_sensorspace_clusters(fl_mean_data, P[10], raw, ax, xvect=freq_vect, base=0.5)
qlt.subpanel_label(ax, 'C')
plt.title('{0}_{1}'.format(gmodel.contrast_names[1], fl_contrast_names[0]))

# Female Mean & Male Mean
plt.subplot(2,4,7)
plt.plot(fx[0], gmodel.copes[2, 1, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 1, : ,:].mean(axis=1))

ax = plt.subplot(144)
plt.plot(xf[0], gmodel.copes[2, 2, : ,:].mean(axis=1))
plt.plot(xf[0], gmodel.copes[3, 2, : ,:].mean(axis=1))
plt.plot(xf[0], gmodel.copes[2, 3, : ,:].mean(axis=1))
plt.plot(xf[0], gmodel.copes[3, 3, : ,:].mean(axis=1))
plt.legend(['{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[2], fl_contrast_names[3]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_contrast_names[3])], frameon=False)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')


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

tstat_args2 = tstat_args.copy()
tstat_args2['smooth_dims'] = 0

plt.figure()
for ii in range(2):
    for jj in range(9):
        ind = (jj+1)+ii*9
        print(ind)
        plt.subplot(2, 8, ind)

        #ts = gmodel.get_tstats(**tstat_args)[ii, jj, : ,:]
        ts = glm.fit.get_tstats(gmodel.copes[ii, jj, :, :], gmodel.varcopes[ii, jj, :, :],
                                **tstat_args2)

        plt.plot(freq_vect, ts)
        plt.title(gl_contrast_names[ii] + ' : ' + fl_contrast_names[jj])
        if ii == 1:
            plt.ylim(-10, 10)

#%% ----------------------------

fx = qlt.prep_scaled_freq(0.5, freq_vect,)

plt.figure(figsize=(16, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.35)
plt.subplots_adjust(hspace=0.4, wspace=0.3, top=1, left=0.075, right=0.975)
ii = 0
for jj in range(8):
    ind = (jj+7)+ii*6
    #ax = plt.subplot(3,6,ind)
    ax = plt.subplot(1,8,jj+1)

    if jj == 0:
        qlt.plot_joint_spectrum(ax, gmodel.copes[0, 0, :, :], raw, freq_vect, base=0.5, freqs=[0.5, 9, 24], topo_scale=None)
    else:
        fl_mean_data = deepcopy(data)
        fl_mean_data.data = data.data[:, jj, : ,:]
        qlt.plot_sensorspace_clusters(fl_mean_data, P[jj], raw, ax,
                                      base=0.5, title=to_permute[jj][2],
                                      ylabel='t-stat', thresh=99, xvect=freq_vect)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height*0.65])

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

ax = plt.subplot(4,3,4)
qlt.plot_joint_spectrum(ax, gmodel_null.copes[0, 0, :, :], raw, freq_vect,
                        title='Group Mean - Null Model', base=0.5,
                        freqs=[1, 9, 24], topo_scale=None)
qlt.subpanel_label(ax, chr(65), yf=1.1)

fl_mean_data = deepcopy(datanull)
fl_mean_data.data = datanull.data[:, 0, : ,:]
ax = plt.subplot(4,3,5)
qlt.plot_sensorspace_clusters(fl_mean_data, Pn[1], raw, ax,
                              base=0.5, title=to_permute_null[1][2],
                              ylabel='t-stat', thresh=95, xvect=freq_vect)
qlt.subpanel_label(ax, chr(66), yf=1.1)
ax = plt.subplot(4,3,6)
qlt.plot_sensorspace_clusters(fl_mean_data, Pn[2], raw, ax,
                              base=0.5, title=to_permute_null[2][2],
                              ylabel='t-stat', thresh=95, xvect=freq_vect)
qlt.subpanel_label(ax, chr(67), yf=1.1)

ax = plt.subplot(4,3,10)
qlt.plot_joint_spectrum(ax, gmodel.copes[0, 0, :, :], raw, freq_vect,
                        title='Group Mean - Full Model', base=0.5,
                        freqs=[1, 9, 24], topo_scale=None)
qlt.subpanel_label(ax, chr(68), yf=1.1)

fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 0, : ,:]
ax = plt.subplot(4,3,11)
qlt.plot_sensorspace_clusters(fl_mean_data, P[6], raw, ax,
                              base=0.5, title=to_permute[6][2],
                              ylabel='t-stat', thresh=95, xvect=freq_vect)
qlt.subpanel_label(ax, chr(69), yf=1.1)
ax = plt.subplot(4,3,12)
qlt.plot_sensorspace_clusters(fl_mean_data, P[7], raw, ax,
                              base=0.5, title=to_permute[7][2],
                              ylabel='t-stat', thresh=95, xvect=freq_vect)
qlt.subpanel_label(ax, chr(70), yf=1.1)

fout = os.path.join(cfg['lemon_analysis_dir'], 'lemon-group_glm-higherordereffects.png')
plt.savefig(fout, transparent=True, dpi=300)


#%% ----------------------------------------


ch = 28
plt.figure()
plt.subplot(121)
plt.plot(freq_vect, datanull.data[males, 1, :, ch].mean(axis=(0)), label='MaleOpen<Closed', lw=2)
plt.plot(freq_vect, datanull.data[females, 1, :, ch].mean(axis=(0)), label='FemaleOpen<Closed', lw=2)
plt.plot(freq_vect, datanull.data[males, 2, :, ch].mean(axis=(0)), label='MaleOpen', lw=2)
plt.plot(freq_vect, datanull.data[females, 2, :, ch].mean(axis=(0)), label='FemaleOpen', lw=2)
plt.plot(freq_vect, datanull.data[males, 3, :, ch].mean(axis=(0)), label='MaleClosed', lw=2)
plt.plot(freq_vect, datanull.data[females, 3, :, ch].mean(axis=(0)), label='FemaleClosed', lw=2)
plt.legend()
plt.subplot(122)
plt.plot(freq_vect, data.data[males, 1, :, ch].mean(axis=(0)), label='MaleOpen<Closed', lw=2)
plt.plot(freq_vect, data.data[females, 1, :, ch].mean(axis=(0)), label='FemaleOpen<Closed', lw=2)
plt.plot(freq_vect, data.data[males, 2, :, ch].mean(axis=(0)), label='MaleOpen', lw=2)
plt.plot(freq_vect, data.data[females, 2, :, ch].mean(axis=(0)), label='FemaleOpen', lw=2)
plt.plot(freq_vect, data.data[males, 3, :, ch].mean(axis=(0)), label='MaleClosed', lw=2)
plt.plot(freq_vect, data.data[females, 3, :, ch].mean(axis=(0)), label='FemaleClosed', lw=2)
plt.legend()

plt.figure()
plt.subplot(121)
plt.plot(freq_vect, datanull.data[males, 1, :, :].mean(axis=(0, 2)), label='MaleOpen<Closed', lw=2)
plt.plot(freq_vect, datanull.data[females, 1, :, :].mean(axis=(0, 2)), label='FemaleOpen<Closed', lw=2)
plt.plot(freq_vect, datanull.data[males, 2, :, :].mean(axis=(0, 2)), label='MaleOpen', lw=2)
plt.plot(freq_vect, datanull.data[females, 2, :, :].mean(axis=(0, 2)), label='FemaleOpen', lw=2)
plt.plot(freq_vect, datanull.data[males, 3, :, :].mean(axis=(0, 2)), label='MaleClosed', lw=2)
plt.plot(freq_vect, datanull.data[females, 3, :, :].mean(axis=(0, 2)), label='FemaleClosed', lw=2)
plt.legend()
plt.subplot(122)
plt.plot(freq_vect, data.data[males, 1, :, :].mean(axis=(0, 2)), label='MaleOpen<Closed', lw=2)
plt.plot(freq_vect, data.data[females, 1, :, :].mean(axis=(0, 2)), label='FemaleOpen<Closed', lw=2)
plt.plot(freq_vect, data.data[males, 2, :, :].mean(axis=(0, 2)), label='MaleOpen', lw=2)
plt.plot(freq_vect, data.data[females, 2, :, :].mean(axis=(0, 2)), label='FemaleOpen', lw=2)
plt.plot(freq_vect, data.data[males, 3, :, :].mean(axis=(0, 2)), label='MaleClosed', lw=2)
plt.plot(freq_vect, data.data[females, 3, :, :].mean(axis=(0, 2)), label='FemaleClosed', lw=2)
plt.legend()







def _get_sensible_ticks(vmin, vmax, nbins=3):
    """Return sensibly rounded tick positions based on a plotting range.

    Based on code in matplotlib.ticker
    Assuming symmetrical axes and 3 ticks for the moment

    """
    scale, offset = ticker.scale_range(vmin, vmax)
    if vmax/scale > 0.5:
        scale = scale / 2
    edge = ticker._Edge_integer(scale, offset)
    low = edge.ge(vmin)
    high = edge.le(vmax)

    ticks = np.linspace(low, high, nbins) * scale

    return ticks


def test_topos(psd):

    plt.figure(figsize=(12,8))
    vmin = psd.min()
    vmax = psd.max()
    if np.all(np.sign((vmin, vmax))==1):
        cmap = 'Reds'
        norm = None
    elif np.all(np.sign((vmin, vmax))==-1):
        cmap = 'Blues'
        norm = None
    elif np.all(np.sign((-vmin, vmax))==1):
        cmap = 'RdBu_r'
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        if np.abs(vmin) < vmax:
            vmin = vmin if np.abs(vmin) > vmax/5 else -vmax/5
        elif np.abs(vmin) > vmax:
            vmax = vmax if np.abs(vmin)/5 < vmax else np.abs(vmin)/5

    plt.subplot(2,1,1)
    plot(psd)

    for idx, ii in enumerate(np.arange(5)*10):
        plt.subplot(2,5,idx+6)
        im,cn = mne.viz.plot_topomap(psd[ii, :], raw.info, vmin=vmin, vmax=vmax, cmap=cmap)
        if norm is not None:
            im.set_norm(norm)
        if idx == 4:
            cb = plt.colorbar(im, boundaries=np.linspace(vmin, vmax))
            tks = _get_sensible_ticks(vmin, vmax, 7)
            cb.set_ticks(tks)








plt.figure()
plt.plot(freq_vect, data.data[data.info['sex']==1, 2, :, :].mean(axis=2).T, 'k')
plt.plot(ff[0, data.info['sex']==1], ff[1, data.info['sex']==1], 'or', alpha=1/3)



plt.figure()
for ii in range(113):
    f, p  = get_peaks(freq_vect, data.data[ii, 3, :, :].mean(axis=1), frange=(7,15))
    plt.plot(freq_vect, data.data[ii, 3, :, :].mean(axis=1), 'k')
    print(f)
    plt.plot(f+0.5, p, 'or')
