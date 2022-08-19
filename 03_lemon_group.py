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

import lemon_plotting
from matplotlib.patches import ConnectionPatch

from glm_config import cfg

#%% ---------------------------------------------------
# Load single subject for reference

fbase = os.path.join(cfg['lemon_processed_data'], 'sub-010002_preproc_raw.fif')
raw = mne.io.read_raw_fif(fbase).pick_types(eeg=True)

st = osl.utils.Study(os.path.join(cfg['lemon_glm_data'],'{subj}_preproc_raw_glm-data.hdf5'))
freq_vect = h5py.File(st.match_files[0], 'r')['freq_vect'][()]
fl_model = obj_from_hdf5file(st.match_files[0], 'model')


#%% --------------------------------------------------
# Load first level results and fit group model

inputs = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_groupdata.hdf5')
nulls = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_groupdata_null.hdf5')

data = obj_from_hdf5file(inputs, 'data')
datanull = obj_from_hdf5file(nulls, 'data')

data.info['age_group'] = np.array(data.info['age']) < 45

# Drop obvious outliers
bads = sails.utils.detect_artefacts(data.data[:, 0, :, :], axis=0)
clean_data = data.drop(np.where(bads)[0])

DC = glm.design.DesignConfig()
DC.add_regressor(name='Young', rtype='Categorical', datainfo='age_group', codes=1)
DC.add_regressor(name='Old', rtype='Categorical', datainfo='age_group', codes=0)
DC.add_regressor(name='Sex', rtype='Parametric', datainfo='sex', preproc='z')
DC.add_contrast(name='Mean',values={'Young': 0.5, 'Old': 0.5})
DC.add_contrast(name='Young>Old',values={'Young': 1, 'Old': -1})
DC.add_simple_contrasts()

design = DC.design_from_datainfo(data.info)
gmodel = glm.fit.OLSModel(design, data)
gmodel_null = glm.fit.OLSModel(design, datanull)


with h5py.File(os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-data.hdf5'), 'w') as F:
     gmodel.to_hdf5(F.create_group('model'))
     design.to_hdf5(F.create_group('design'))
     data.to_hdf5(F.create_group('data'))

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_glm-design.png')
design.plot_summary(show=False, savepath=fout)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_glm-efficiency.png')
design.plot_efficiency(show=False, savepath=fout)


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
              (1, 0, 'Group Effect of Age on Mean'),
              (1, 1, 'Group Effect of Age on Open>Closed'),
              (1, 2, 'Group Effect of Age on Open'),
              (1, 3, 'Group Effect of Age on Closed')]


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

        p = glm.permutations.MNEClusterPermutation(design, fl_mean_data, gl_con, 250,
                                                   nprocesses=8,
                                                   metric='tstats',
                                                   cluster_forming_threshold=cft,
                                                   tstat_args=tstat_args,
                                                   adjacency=adjacency)

        with open(os.path.join(cfg['lemon_glm_data'], 'lemon-group_perms-con{0}.pkl'.format(icon)), "wb") as dill_file:
            dill.dump(p, dill_file)

        P.append(p)
    else:
        dill_file = os.path.join(cfg['lemon_glm_data'], 'lemon-group_perms-con{0}.pkl'.format(icon))
        P.append(dill.load(open(dill_file, 'rb')))

# Permuate
# Blinks on Mean, Mean on linear, task, blinks, bads
to_permute_null = [(0, 0, 'Overall Mean'),
                   (0, 1, 'Group Mean of Open>Closed'),
                   (0, 2, 'Group Mean of Open'),
                   (0, 3, 'Group Mean of Closed'),
                   (1, 0, 'Group Effect of Age on Mean'),
                   (1, 1, 'Group Effect of Age on Open>Closed'),
                   (1, 2, 'Group Effect of Age on Open'),
                   (1, 3, 'Group Effect of Age on Closed')]

Pn = []

run_perms = False
for icon in range(len(to_permute_null)):
    if to_permute_null[icon][1] > 3:
        Pn.append(None)
    elif run_perms:
        gl_con = to_permute_null[icon][0]
        fl_con = to_permute_null[icon][1]
        # Only working with mean regressor for the moment
        fl_mean_data = deepcopy(datanull)
        fl_mean_data.data = datanull.data[:, fl_con, : ,:]

        p = glm.permutations.MNEClusterPermutation(design, fl_mean_data, gl_con, 250,
                                                   nprocesses=8,
                                                   metric='tstats',
                                                   cluster_forming_threshold=cft,
                                                   tstat_args=tstat_args,
                                                   adjacency=adjacency)

        with open(os.path.join(cfg['lemon_glm_data'], 'lemon-group_permsnull-con{0}.pkl'.format(icon)), "wb") as dill_file:
            dill.dump(p, dill_file)

        Pn.append(p)
    else:
        dill_file = os.path.join(cfg['lemon_glm_data'], 'lemon-group_permsnull-con{0}.pkl'.format(icon))
        Pn.append(dill.load(open(dill_file, 'rb')))



#%% ----------------------------


## SHOULD REPLACE WITH GLM.VIZ version
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

#%% -----------------------------------------------------

sensor = 'Pz'
ch_ind = mne.pick_channels(raw.ch_names, [sensor])[0]

I = np.argsort(data.data[:, 0, :, 23].sum(axis=1))
I = np.arange(48)

plt.figure(figsize=(16, 9))
aspect = 16/9
xf = lemon_plotting.prep_scaled_freq(0.5, freq_vect)

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
    d = data.data[I[ii],0, :, ch_ind]
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
lemon_plotting.subpanel_label(subj_ax, chr(65), yf=0.75, xf=0.05)
subj_ax.text(0.125, 0.725, 'First Level GLM Spectra', transform=subj_ax.transAxes, fontsize='large')

pcm = plot_design(des_ax, design.design_matrix[:, :], design.regressor_names)
des_ax.set_xticklabels(['Young', 'Old', 'Sex', ''])
des_ax.set_ylabel('Participants')
des_ax.set_title('Group Design Matrix')
lemon_plotting.subpanel_label(des_ax, chr(65+1), yf=1.05)
plt.colorbar(pcm, cax=cb_dm)

mean_ax.errorbar(xf[0], gmodel.copes[0, 0, :, ch_ind], yerr=np.sqrt(gmodel.varcopes[0, 0, :, ch_ind]), errorevery=1)
mean_ax.set_xticks(xf[2], xf[1])
mean_ax.set_title('Group Mean Spectrum')
lemon_plotting.decorate_spectrum(mean_ax, ylabel='Amplitude')
lemon_plotting.subpanel_label(mean_ax, chr(65+2), yf=1.1)
mean_ax.set_xlim(0)
mean_ax.set_ylim(0)

cov_ax.errorbar(xf[0], gmodel.copes[1, 0, :, ch_ind], yerr=np.sqrt(gmodel.varcopes[2, 0, :, ch_ind]), errorevery=2)
cov_ax.errorbar(xf[0], gmodel.copes[4, 0, :, ch_ind], yerr=np.sqrt(gmodel.varcopes[3, 0, :, ch_ind]), errorevery=2)
cov_ax.set_title('Group effects on Mean Spectrum')
cov_ax.legend(list(np.array(gmodel.contrast_names)[[1,4]]))
cov_ax.set_xticks(xf[2], xf[1])
lemon_plotting.decorate_spectrum(cov_ax, ylabel='Amplitude')
lemon_plotting.subpanel_label(cov_ax, chr(65+3), yf=1.1)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_glm-overview.png')
plt.savefig(fout, transparent=True, dpi=300)


#%% ----------------------------
fx = lemon_plotting.prep_scaled_freq(0.5, freq_vect,)

Q = P

plt.figure(figsize=(16, 12))

ax = plt.axes((0.075, 0.675, 0.3, 0.25))
lemon_plotting.plot_joint_spectrum(ax, gmodel.copes[0, 2, :, :], raw, freq_vect, base=0.5, freqs=[10, 22],
                        topo_scale=None, title='GroupMean Eyes Open', ylabel='FFT Magnitude')
lemon_plotting.subpanel_label(ax, 'A')

ax = plt.axes((0.45, 0.675, 0.3, 0.25))
lemon_plotting.plot_joint_spectrum(ax, gmodel.copes[0, 3, :, :], raw, freq_vect, base=0.5, freqs=[10, 22],
                        topo_scale=None, title='GroupMean Eyes Closed', ylabel='FFT Magnitude')
lemon_plotting.subpanel_label(ax, 'B')

ax = plt.axes([0.775, 0.7, 0.15, 0.15])
lemon_plotting.plot_channel_layout(ax, raw, size=100)

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

    lemon_plotting.plot_sensorspace_clusters(fl_mean_data, P[ii+4], raw, ax, xvect=freq_vect, base=0.5)
    ax.set_ylabel('t-stat')
    lemon_plotting.subpanel_label(ax, chr(65+ii+2))
    plt.title('Group Mean of {0}'.format(fl_model.contrast_names[ii+4]))

    ax = plt.subplot(4,5,ii+16)

    beta0 = gmodel.betas[:, 2:4, :, :].mean(axis=(0,1))
    beta1 = gmodel.betas[:, ii+4, :, :].mean(axis=0)

    if ii == 0:
        ax.plot(fx[0], np.mean(beta0 + -1*beta1, axis=1))
    else:
        ax.plot(fx[0], np.mean(beta0 + 0*beta1, axis=1))
    ax.plot(fx[0], np.mean(beta0 + 1*beta1, axis=1))
    ax.set_xticks(fx[2], fx[1])
    lemon_plotting.decorate_spectrum(ax)

    plt.legend(ll[ii], frameon=False)


fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-meancov.png')
plt.savefig(fout, transparent=True, dpi=300)



#%% ----------------------------

plt.figure(figsize=(16, 9))

# Open mean & Closed Mean
ax = plt.axes([0.075, 0.5, 0.175, 0.4])
fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 1, : ,:]  # Open>Closed
lemon_plotting.plot_sensorspace_clusters(fl_mean_data, P[1], raw, ax, xvect=freq_vect, base=0.5)
ax.set_ylabel('t-stat')
lemon_plotting.subpanel_label(ax, 'A')
plt.title('Within Subject Effect\n{0}_{1}'.format(gmodel.contrast_names[0], fl_model.contrast_names[1]))

ax = plt.axes([0.075, 0.1, 0.175, 0.30])
plt.plot(fx[0], gmodel.copes[0, 2, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[0, 3, : ,:].mean(axis=1))
lemon_plotting.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
plt.legend(['Eyes Open', 'Eyes Closed'], frameon=False)


# Young and Old Means
ax = plt.axes([0.3125, 0.5, 0.175, 0.4])
fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 0, : ,:]  # Mean
lemon_plotting.plot_sensorspace_clusters(fl_mean_data, P[9], raw, ax, xvect=freq_vect, base=0.5)
ax.set_ylabel('t-stat')
lemon_plotting.subpanel_label(ax, 'B')
plt.title('Between Subject Effect\n{0}_{1}'.format(gmodel.contrast_names[1], fl_model.contrast_names[0]))

ax = plt.axes([0.3125, 0.1, 0.175, 0.3])
plt.plot(fx[0], gmodel.copes[2, 0, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 0, : ,:].mean(axis=1))
lemon_plotting.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
plt.legend(['Young', 'Old'], frameon=False)


# Interactions
ax = plt.axes([0.55, 0.5, 0.175, 0.4])
fl_mean_data = deepcopy(data)
fl_mean_data.data = data.data[:, 1, : ,:]  # Open>Clonsed
lemon_plotting.plot_sensorspace_clusters(fl_mean_data, P[10], raw, ax, xvect=freq_vect, base=0.5)
ax.set_ylabel('t-stat')
lemon_plotting.subpanel_label(ax, 'C')
plt.title('Interaction Effect\n{0}_{1}'.format(gmodel.contrast_names[1], fl_model.contrast_names[1]))

ax = plt.axes([0.55, 0.1, 0.175, 0.3])
plt.plot(fx[0], gmodel.copes[2, 1, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 1, : ,:].mean(axis=1))
lemon_plotting.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
plt.legend(['Young : Open > Closed', 'Old : Open > Closed'], frameon=False)

ax = plt.axes([0.775, 0.7, 0.2, 0.2])
lemon_plotting.plot_channel_layout(ax, raw, size=100)

#ax = plt.subplot(4,4,(12,16))
ax = plt.axes([0.8, 0.175, 0.175, 0.35])
plt.plot(fx[0], gmodel.copes[2, 2, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 2, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[2, 3, : ,:].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 3, : ,:].mean(axis=1))
plt.legend(['{0}_{1}'.format(gmodel.contrast_names[2], fl_model.contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_model.contrast_names[2]),
            '{0}_{1}'.format(gmodel.contrast_names[2], fl_model.contrast_names[3]),
            '{0}_{1}'.format(gmodel.contrast_names[3], fl_model.contrast_names[3])], frameon=False)
lemon_plotting.decorate_spectrum(ax, ylabel='FFT Magnitude')
ax.set_xticks(fx[2], fx[1])
lemon_plotting.subpanel_label(ax, 'D')

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-anova.png')
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

plt.figure(figsize=(16, 9))
ax_eo = plt.axes([0.1, 0.35, 0.25, 0.55])
ax_eo_f = plt.axes([0.1, 0.075, 0.25, 0.2])
ax_eo_a = plt.axes([0.375, 0.35, 0.1, 0.55])

ax_ec = plt.axes([0.55, 0.35, 0.25, 0.55])
ax_ec_f = plt.axes([0.55, 0.075, 0.25, 0.2])
ax_ec_a = plt.axes([0.825, 0.35, 0.1, 0.55])

ff = np.zeros((2, 202))
for ii in range(202):
    ff[:, ii] = get_peaks(freq_vect, data.data[ii, 2, :, :].mean(axis=1), frange=(6,15))

ax_eo.plot(ff[0, data.info['age_group']==True], ff[1, data.info['age_group']==True], 'or', alpha=1/3, mec='white')
ax_eo.plot(ff[0, data.info['age_group']==False], ff[1, data.info['age_group']==False], 'ob', alpha=1/3, mec='white')
ax_eo.plot(freq_vect, data.data[data.info['age_group']==True, 2, :, :].mean(axis=(0, 2)), 'r', lw=3)
ax_eo.plot(freq_vect, data.data[data.info['age_group']==False, 2, :, :].mean(axis=(0, 2)), 'b', lw=3)
ax_eo.set_xlim(0, 20)
ax_eo.set_ylim(0, 1.2e-5)
l = plt.legend(['Young Subject Peak', 'Old Subject Peak', 'Young Group Spectrum', 'Old Group Spectrum'], bbox_to_anchor=(1, 1))
lemon_plotting.decorate_spectrum(ax_eo, ylabel='FFT Magnitude')
ax_eo.set_title('Eyes Open Rest')
lemon_plotting.subpanel_label(ax_eo, 'A')
ax_eo.set_xticks(np.linspace(0,20,5))

yy = ff[0, data.info['age_group']==False]
yy = yy[np.logical_not(np.isnan(yy))]
ax_eo_f.boxplot(yy, vert=False, positions=[0.8], **blue_dict)
xx = ff[0, data.info['age_group']==True]
xx = xx[np.logical_not(np.isnan(xx))]
ax_eo_f.boxplot(xx, vert=False, positions=[1], **red_dict)
ax_eo_f.set_xlim(6, 14)
ax_eo_f.set_ylim(0.6, 1.2)
ax_eo_f.set_yticklabels(('Old', 'Young'))
for tag in ['top', 'right']:
    ax_eo_f.spines[tag].set_visible(False)
ax_eo_f.spines['left'].set_bounds(0.7, 1.1)
ax_eo_f.set_xlabel('Frequency (Hz)')
lemon_plotting.subpanel_label(ax_eo_f, 'B')

#con1 = ConnectionPatch(xyA=(6, 0), xyB=(6, 1.1), coordsA="data", coordsB="data",
#                      axesA=ax_eo, axesB=ax_eo_f, color="black")
#ax_eo.add_artist(con1)
#con2 = ConnectionPatch(xyA=(14, 0), xyB=(14, 1.1), coordsA="data", coordsB="data",
#                      axesA=ax_eo, axesB=ax_eo_f, color="black")
#ax_eo.add_artist(con2)

yy = ff[1, data.info['age_group']==False]
yy = yy[np.logical_not(np.isnan(yy))]
ax_eo_a.boxplot(yy, vert=True, positions=[0.8], **blue_dict)
xx = ff[1, data.info['age_group']==True]
xx = xx[np.logical_not(np.isnan(xx))]
ax_eo_a.boxplot(xx, vert=True, positions=[1], **red_dict)
ax_eo_a.set_ylim(0, 1.2e-5)
ax_eo_a.set_xlim(0.6, 1.2)
ax_eo_a.set_xticklabels(('Old', 'Young'))
for tag in ['top', 'right']:
    ax_eo_a.spines[tag].set_visible(False)
lemon_plotting.subpanel_label(ax_eo_a, 'C')

tt = stats.ttest_ind(xx, yy)
print('Eyes Open')
print('Male mean : {}'.format(yy.mean()))
print('Female mean : {}'.format(xx.mean()))
print('t({0})={1}, p={2}'.format(len(xx)+len(yy)-2, tt.statistic, tt.pvalue))

ff = np.zeros((2, 202))
for ii in range(202):
    ff[:, ii] = get_peaks(freq_vect, data.data[ii, 3, :, :].mean(axis=1), frange=(6,15))

ax_ec.plot(ff[0, data.info['age_group']==True], ff[1, data.info['age_group']==True], 'or', alpha=1/3, mec='white')
ax_ec.plot(ff[0, data.info['age_group']==False], ff[1, data.info['age_group']==False], 'ob', alpha=1/3, mec='white')
ax_ec.plot(freq_vect, data.data[data.info['age_group']==True, 3, :, :].mean(axis=(0, 2)), 'r', lw=3)
ax_ec.plot(freq_vect, data.data[data.info['age_group']==False, 3, :, :].mean(axis=(0, 2)), 'b', lw=3)
ax_ec.set_xlim(0, 20)
ax_ec.set_ylim(0, 1.2e-5)
lemon_plotting.decorate_spectrum(ax_ec, ylabel='FFT Magnitude')
ax_ec.set_title('Eyes Closed Rest')
lemon_plotting.subpanel_label(ax_ec, 'D')
ax_ec.set_xticks(np.linspace(0,20,5))

yy = ff[0, data.info['age_group']==False]
yy = yy[np.logical_not(np.isnan(yy))]
ax_ec_f.boxplot(yy, vert=False, positions=[0.8], **blue_dict)
xx = ff[0, data.info['age_group']==True]
xx = xx[np.logical_not(np.isnan(xx))]
ax_ec_f.boxplot(xx, vert=False, positions=[1], **red_dict)
ax_ec_f.set_xlim(6, 14)
ax_ec_f.set_ylim(0.6, 1.2)
ax_ec_f.set_yticklabels(('Old', 'Young'))
for tag in ['top', 'right']:
    ax_ec_f.spines[tag].set_visible(False)
ax_ec_f.spines['left'].set_bounds(0.7, 1.1)
ax_ec_f.set_xlabel('Frequency (Hz)')
lemon_plotting.subpanel_label(ax_ec_f, 'E')

#con3 = ConnectionPatch(xyA=(6, 0), xyB=(6, 1.1), coordsA="data", coordsB="data",
#                      axesA=ax_ec, axesB=ax_ec_f, color="black")
#ax_ec.add_artist(con3)
#con4 = ConnectionPatch(xyA=(14, 0), xyB=(14, 1.1), coordsA="data", coordsB="data",
#                      axesA=ax_ec, axesB=ax_ec_f, color="black")
#ax_ec.add_artist(con4)

yy = ff[1, data.info['age_group']==False]
yy = yy[np.logical_not(np.isnan(yy))]
ax_ec_a.boxplot(yy, vert=True, positions=[0.8], **blue_dict)
xx = ff[1, data.info['age_group']==True]
xx = xx[np.logical_not(np.isnan(xx))]
ax_ec_a.boxplot(xx, vert=True, positions=[1], **red_dict)
ax_ec_a.set_ylim(0, 1.2e-5)
ax_ec_a.set_xlim(0.6, 1.2)
ax_ec_a.set_xticklabels(('Old', 'Young'))
for tag in ['top', 'right']:
    ax_ec_a.spines[tag].set_visible(False)
lemon_plotting.subpanel_label(ax_ec_a, 'F')

tt = stats.ttest_ind(xx, yy)
print('Eyes Closed')
print('Male mean : {}'.format(yy.mean()))
print('Female mean : {}'.format(xx.mean()))
print('t({0})={1}, p={2}'.format(len(xx)+len(yy)-2, tt.statistic, tt.pvalue))

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-age-peaks.png')
plt.savefig(fout, transparent=True, dpi=300)