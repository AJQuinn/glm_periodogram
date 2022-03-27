import os
import sys
import osl
import numpy as np
import mne
import sails
import pprint
from scipy import io, ndimage
import matplotlib.pyplot as plt

sys.path.append('/Users/andrew/src/qlt')
import qlt

import logging
logger = logging.getLogger('osl')

figbase = '/Users/andrew/Projects/glm/glm_psd/figures/'
outdir = '/Users/andrew/Projects/glm/glm_psd/analysis'

#%% --------------------------------------------------
# Preprocessing

from lemon_support import (lemon_make_blinks_regressor,
                           lemon_make_task_regressor,
                           lemon_set_channel_montage, lemon_ica)

config = osl.preprocessing.load_config('lemon_preproc.yml')
pprint.pprint(config)

outf = os.path.join(figbase, 'EEG_Lemon_preproc_flowchart.png')
osl.preprocessing.plot_preproc_flowchart(config, outname=outf, show=False, stagecol='lightskyblue', startcol='coral')

# Prep
subj = '010060'
base = f'/Users/andrew/Projects/lemon/EEG_Raw_BIDS_ID/sub-{subj}/RSEEG'
infile = os.path.join(base, f'sub-{subj}.vhdr')
extras = [lemon_set_channel_montage, lemon_ica]

# Run preproc with and without ICA
dataset_ica = osl.preprocessing.run_proc_chain(infile, config, extra_funcs=extras)
print(config['preproc'][8])
del config['preproc'][8]  # Drop ICA stage
pprint.pprint(config)
dataset_noica = osl.preprocessing.run_proc_chain(infile, config, extra_funcs=extras)

#%% ----------------------------------------------------------
# GLM-Prep

# Make blink regressor
blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(dataset_noica['raw'])

# Make task regressor
task = lemon_make_task_regressor(dataset_noica)

#%% ----------------------------------------------------------
# Now run four models
# 1 - no-ica - no confound
# 2 - no-ica + confound
# 3 - ica - no confound
# 4 - ica + confound

fs = int(dataset_noica['raw'].info['sfreq'])
inds = np.arange(247000, 396000) + fs*60  # about a minute of eyes open
inds = Ellipsis
sensor = 'Pz'
XX = dataset_noica['raw'].get_data(picks=sensor)[:, inds].T
f, copes, varcopes, extras1 = sails.stft.glm_periodogram(XX, axis=0,
                                                         covariates={'Open>Closed': task},
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs,
                                                         fit_method='glmtools')
f, copes, varcopes, extras2 = sails.stft.glm_periodogram(XX, axis=0,
                                                         covariates={'Open>Closed': task},
                                                         confounds={'blinks': blink_vect[inds]},
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs,
                                                         fit_method='glmtools')
XX = dataset_ica['raw'].get_data(picks=sensor)[:, inds].T
f, copes, varcopes, extras3 = sails.stft.glm_periodogram(XX, axis=0,
                                                         covariates={'Open>Closed': task},
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs,
                                                         fit_method='glmtools')
f, copes, varcopes, extras4 = sails.stft.glm_periodogram(XX, axis=0,
                                                         covariates={'Open>Closed': task},
                                                         confounds={'blinks': blink_vect[inds]},
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs,
                                                         fit_method='glmtools')


#%% ------------------------------------------------------------------
# Prep for figure 2

def plot_design(ax, design_matrix, regressor_names):
    num_observations, num_regressors = design_matrix.shape
    vm = np.max((design_matrix.min(), design_matrix.max()))
    cax = ax.pcolor(design_matrix, cmap=plt.cm.coolwarm,
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


def subpanel_label(ax, label, xf=-0.1, yf=1.1):
    ypos = ax.get_ylim()[0]
    yyrange = np.diff(ax.get_ylim())[0]
    ypos = (yyrange * yf) + ypos
    # Compute letter position as proportion of full xrange.
    xpos = ax.get_xlim()[0]
    xxrange = np.diff(ax.get_xlim())[0]
    xpos = (xxrange * xf) + xpos
    ax.text(xpos, ypos, label, horizontalalignment='center',
            verticalalignment='center', fontsize=20, fontweight='bold')


def decorate(ax):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)


# Extract data segment
inds = np.arange(140*fs, 160*fs)
XX = dataset_ica['raw'].get_data(picks=sensor)[:, inds].T
mini_task = task[inds]
eog = dataset_ica['raw'].copy().pick_types(eog=True, eeg=False)
eog.filter(l_freq=1, h_freq=25, picks='eog')
eog = eog.get_data(picks='eog')[:, inds].T
time = np.linspace(0, XX.shape[0]/fs, XX.shape[0])

# Compute STFT for data segment
config = sails.stft.PeriodogramConfig(input_len=XX.shape[0],
                                      fs=fs, nperseg=fs*2,
                                      axis=0, fmin=0.1,  fmax=45)
fw, tw, stft = sails.stft.compute_stft(XX[:, 0], **config.stft_args)

# Compute mini-model to get design matrix
f, copes, varcopes, extras5 = sails.stft.glm_periodogram(XX, axis=0,
                                                         covariates={'Open>Closed': mini_task},
                                                         confounds={'Blinks': blink_vect[inds]},
                                                         nperseg=fs*2,
                                                         fmin=0.1, fmax=45,
                                                         fs=fs,
                                                         fit_method='glmtools')
model, design, data = extras5

# prep sqrt(f) axes
fx, ftl, ft = qlt.prep_scaled_freq(0.5, f)

#%% ------------------------------------------------------------
# Make figure 2

model, design, data = extras5
plt.figure(figsize=(16, 9))

ax_ts = plt.axes([0.05, 0.1, 0.066, 0.8])
ax_tf = plt.axes([0.13, 0.1, 0.125, 0.8])
cb_tf = plt.axes([0.25, 0.15, 0.01, 0.2])

ax_cov = plt.axes([0.4, 0.1, 0.066, 0.8])
ax_dm = plt.axes([0.48, 0.1, 0.125, 0.8])
cb_dm = plt.axes([0.61, 0.15, 0.01, 0.2])

ax_beta_avg = plt.axes([0.725, 0.5, 0.3*0.8, 0.35*0.8])
ax_beta_cov = plt.axes([0.725, 0.1, 0.3*0.8, 0.35*0.8])

ax_ts.plot(XX-XX.mean(), time, 'k', linewidth=2/3)
ax_ts.set_ylim(time[0], time[-1])
ax_ts.set_ylabel('Time (s)')
ax_ts.text(0, 316.1 ,r'Data', va='bottom', ha='center')
ax_ts.set_title('EEG Fz')
qlt.subpanel_label(ax_ts, 'A', yf=1.04); decorate(ax_ts)

pcm = ax_tf.pcolormesh(fx, tw, np.abs(stft), shading='nearest', cmap='hot_r')
ax_tf.set_ylim(time[0], time[-1])
ax_tf.set_xticks(ft)
ax_tf.set_xticklabels(ftl)
ax_tf.set_yticklabels([])
ax_tf.set_xlim(0, 7)
ax_tf.set_xlabel('Frequency (Hz)')
ax_tf.set_title('Short Time\nFourier Transform')
qlt.subpanel_label(ax_tf, 'B', yf=1.04); decorate(ax_tf)
ax_tf.text(30, 316.1 ,r'Short Time\\Fourier Transform: $Y(f)$', va='bottom', ha='center')
plt.colorbar(pcm, cax=cb_tf, label='Power $(x10^4)$')

ax_cov.plot(eog*1e5, time, 'blue', linewidth=2)
ax_cov.set_ylim(time[0], time[-1])
ax_cov.set_ylabel('Time (s)')
ax_cov.set_xlabel('EOG')
ax_cov.text(10, 316.1 ,r'Covariate', va='bottom', ha='center')
ax_cov.set_title('EOG')
qlt.subpanel_label(ax_cov, 'C', yf=1.04); decorate(ax_cov)

pcm = plot_design(ax_dm, design.design_matrix, design.regressor_names)
ax_dm.set_ylim(0, 19)  # normally 19 windows
ax_dm.set_yticklabels([])
ax_dm.set_xticks([0.5, 1.5, 2.5])
ax_dm.set_xticklabels(['Mean', 'Open>Closed', 'Blinks'])
ax_dm.set_xlabel('Regressors')
ax_dm.set_title('Design Matrix')
qlt.subpanel_label(ax_dm, 'D', yf=1.04); decorate(ax_dm)
ax_dm.text(0.5, 316.1 ,r'Design Matrix: $X$', va='bottom', ha='center')
plt.colorbar(pcm, cax=cb_dm)

model = extras2[0]
ax_beta_avg.errorbar(fx, model.copes[0, :, 0], yerr=np.sqrt(model.varcopes[0, :, 0]), errorevery=2)
ax_beta_avg.legend(model.contrast_names, frameon=False)
ax_beta_avg.set_xticks(ft)
ax_beta_avg.set_xticklabels(ftl)
ax_beta_avg.set_xlim(fx[0], fx[-1])
ax_beta_avg.set_xlabel('Frequency (Hz)')
ax_beta_avg.set_ylabel(r'$\beta$')
ax_beta_avg.text(20, 3.5e-11 ,r'$\hat{Y}(f) = \beta(f)X$', ha='center', fontsize=22)
ax_beta_avg.set_title('Mean Regressor\nParameter Estimates')
qlt.subpanel_label(ax_beta_avg, 'E'); decorate(ax_beta_avg)

ax_beta_cov.errorbar(fx, model.copes[1, :, 0], yerr=np.sqrt(model.varcopes[1, :, 0]), errorevery=2)
ax_beta_cov.errorbar(fx, model.copes[2, :, 0], yerr=np.sqrt(model.varcopes[2, :, 0]), errorevery=2)
ax_beta_cov.legend(model.contrast_names[1:], frameon=False)
ax_beta_cov.set_xticks(ft)
ax_beta_cov.set_xticklabels(ftl)
ax_beta_cov.set_xlim(fx[0], fx[-1])
ax_beta_cov.set_xlabel('Frequency (Hz)')
ax_beta_cov.set_ylabel(r'$\beta$')
ax_beta_cov.set_title('Covariate & Confound Regressor\nParameter Estimates')
qlt.subpanel_label(ax_beta_cov, 'F'); decorate(ax_beta_cov)

fout = os.path.join(outdir, 'sub-010060_proc-full_glm-singlechannelsummary.png')
plt.savefig(fout, dpi=300, transparent=True)

#%% --------------------------------------------------------------
# Prep for figure 3

import glmtools as glm

# TASK
spower = 1
P1 = glm.permutations.ClusterPermutation(extras2[1], extras2[2], 1, 500,
                                         pooled_dims=[1],
                                         tstat_args={'sigma_hat': 'auto'},
                                         cluster_forming_threshold=3.2**spower,
                                         metric='tstats', stat_power=spower)
# BLINKS
P2 = glm.permutations.ClusterPermutation(extras2[1], extras2[2], 2, 500,
                                         pooled_dims=[1],
                                         tstat_args={'sigma_hat': 'auto'},
                                         cluster_forming_threshold=3.2**spower,
                                         metric='tstats', stat_power=spower)

def decorate(ax):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)

def nudge_ax(ax, xval=0, yval=0):
    pos = list(ax.get_position().bounds)
    pos[0] = pos[0] + xval
    pos[1] = pos[1] + yval
    ax.set_position(pos)


# prep sqrt(f) axes
fx, ftl, ft = qlt.prep_scaled_freq(0.5, f)

#%% ----------------------------------------------------------------
# Figure 3 - without ICA

shade = [0.7, 0.7, 0.7]
xf = -0.03
plt.figure(figsize=(9, 9))
#plt.subplots_adjust(wspace=0.3, hspace=0.5, right=0.9, left=0.1)
plt.subplots_adjust(hspace=0.4)

titles = ['Task', 'Blink']

for ii in range(2):
    ax1 = plt.subplot(3, 4, 1+ii*2)
    ax1.plot(fx, extras2[0].copes[ii+1, :, 0], color='k')
    qlt.subpanel_label(plt.gca(), chr(65+ii), yf=1.2); decorate(plt.gca());
    ax1.set_xticks(ft)
    ax1.set_xticklabels(ftl)
    ax1.set_xlim(fx[0], fx[-1])
    plt.ylabel('COPE')
    plt.title(' '*8 + '{0} Regressor\n'.format(titles[ii]), fontsize=16)

    ax2 = plt.subplot(3, 4, 2+ii*2)
    ax2.plot(fx, np.sqrt(extras2[0].varcopes[ii+1, :, 0]), color='r')
    decorate(plt.gca());
    ax2.set_xticks(ft)
    ax2.set_xticklabels(ftl)
    ax2.set_xlim(fx[0], fx[-1])
    plt.ylabel('STDCOPE')

    nudge_ax(ax1, xval=-0.05+0.1*ii)
    nudge_ax(ax2, xval=-0.05+0.1*ii)

    P = P1 if ii == 0 else P2

    ts = extras2[0].get_tstats(sigma_hat='auto')
    ax = plt.subplot(3, 2, 3+ii)
    clu, cstat = P.get_sig_clusters(extras2[2], 99)
    for c in range(len(cstat)):
        inds = np.where(clu==c+1)[0]
        plt.axvspan(fx[inds[0]], fx[inds[-1]], facecolor=shade, alpha=0.5)
    plt.plot(fx, ts[ii+1, :, 0], 'k')
    ax.set_xticks(ft)
    ax.set_xticklabels(ftl)
    ax.set_xlim(fx[0], fx[-1])
    plt.ylabel('t-statistic')
    nudge_ax(ax, xval=-0.05+0.1*ii)
    qlt.subpanel_label(plt.gca(), chr(67+ii), yf=1.2); decorate(plt.gca())

    ax = plt.subplot(3, 2, 5+ii)
    proj, ll =  extras2[0].project_range(ii+1)
    plt.plot(fx, proj[:, :, 0].T)
    if ii == 0:
        plt.legend(['Eyes Closed', 'Eyes Open'], frameon=False)
    else:
        plt.legend(['No Blink', 'Blink'], frameon=False)
    for c in range(len(cstat)):
        inds = np.where(clu==c+1)[0]
        plt.axvspan(fx[inds[0]], fx[inds[-1]], facecolor=shade, alpha=0.5, label=None)
    ax.set_xticks(ft)
    ax.set_xticklabels(ftl)
    ax.set_xlim(fx[0], fx[-1])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (a.u.)')
    nudge_ax(ax, xval=-0.05+0.1*ii)
    qlt.subpanel_label(plt.gca(), chr(69+ii), yf=1.2); decorate(plt.gca())


fout = os.path.join(outdir, 'sub-010060_proc-full_glm-singlechannelstats.png')
plt.savefig(fout, dpi=300, transparent=True)

fout = os.path.join(figbase, 'sub-010060_proc-full_glm-design.png.png')
extras4[1].plot_summary(show=False, savepath=fout)
fout = os.path.join(figbase, 'sub-010060_proc-full_glm-efficiency.png.png')
extras4[1].plot_efficiency(show=False, savepath=fout)


#%% --------------------------------------------------------------------
# Prep for Figure 3 - Supplemental, with ICA

# TASK
spower = 1
P1 = glm.permutations.ClusterPermutation(extras4[1], extras4[2], 1, 500,
                                         pooled_dims=[1],
                                         tstat_args={'sigma_hat': 'auto'},
                                         cluster_forming_threshold=3.2**spower,
                                         metric='tstats', stat_power=spower)
# BLINKS
P2 = glm.permutations.ClusterPermutation(extras4[1], extras4[2], 2, 500,
                                         pooled_dims=[1],
                                         tstat_args={'sigma_hat': 'auto'},
                                         cluster_forming_threshold=3.2**spower,
                                         metric='tstats', stat_power=spower)

def decorate(ax):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    ax.set_xlim(0, 25)

#%% --------------------------------------------------------------------
# Prep for Figure 3 - Supplemental, with ICA

shade = [0.7, 0.7, 0.7]
xf = -0.03
plt.figure(figsize=(12, 9))
plt.subplots_adjust(wspace=0.3, hspace=0.5, right=0.975, left=0.1)

plt.subplot(3, 3, 1)
plt.plot(f, extras4[0].copes[1, :, 0], 'k')
subpanel_label(plt.gca(), 'A', xf=xf); decorate(plt.gca());
plt.ylabel('Parameter Estimate')
plt.title('Task Condition\n', fontsize=16)

ts = extras4[0].get_tstats(sigma_hat='auto')
plt.subplot(3,3,4)
clu, cstat = P1.get_sig_clusters(extras4[2], 99)
for c in range(len(cstat)):
    inds = np.where(clu==c+1)[0]
    plt.axvspan(f[inds[0]], f[inds[-1]], facecolor=shade, alpha=0.5)
plt.plot(f, ts[1, :, 0], 'k')
subpanel_label(plt.gca(), 'B', xf=xf); decorate(plt.gca())
plt.ylabel('t-statistic')

plt.subplot(3,3,7)
proj, ll =  extras4[0].project_range(1)
plt.plot(f, proj[:, :, 0].T)
subpanel_label(plt.gca(), 'C', xf=xf); decorate(plt.gca())
plt.legend(['Eyes Closed', 'Eyes Open'], frameon=False)
for c in range(len(cstat)):
    inds = np.where(clu==c+1)[0]
    plt.axvspan(f[inds[0]], f[inds[-1]], facecolor=shade, alpha=0.5, label=None)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')

plt.subplot(3, 3, 2)
plt.plot(f, extras4[0].copes[2, :, 0], 'k')
subpanel_label(plt.gca(), 'D', xf=xf); decorate(plt.gca())
plt.title('Blinking Confound\n', fontsize=16)

plt.subplot(3,3,5)
clu, cstat = P2.get_sig_clusters(extras4[2], 99)
for c in range(len(cstat)):
    inds = np.where(clu==c+1)[0]
    plt.axvspan(f[inds[0]], f[inds[-1]], facecolor=shade, alpha=0.5)
plt.plot(f, ts[2, :, 0], 'k')
subpanel_label(plt.gca(), 'E', xf=xf); decorate(plt.gca())

plt.subplot(3,3,8)
proj, ll =  extras4[0].project_range(2)
plt.plot(f, proj[:, :, 0].T)
plt.legend(['Blinking', 'No Blinking'], frameon=False)
for c in range(len(cstat)):
    inds = np.where(clu==c+1)[0]
    plt.axvspan(f[inds[0]], f[inds[-1]], facecolor=shade, alpha=0.5)
subpanel_label(plt.gca(), 'F', xf=xf); decorate(plt.gca())
plt.xlabel('Frequency (Hz)')

plt.subplot(3,3,6)
plt.plot(f, extras1[0].copes[0, :, 0])
plt.plot(f, extras4[0].copes[0, :, 0])
subpanel_label(plt.gca(), 'G', xf=xf); decorate(plt.gca())
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')
plt.title('Mean Spectrum from\nboth models', fontsize=16)

fout = os.path.join(figbase, 'EEG_Lemon_singlechannel_stats_ica.png')
plt.savefig(fout, dpi=300, transparent=True)

