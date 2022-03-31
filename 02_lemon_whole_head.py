import os
import osl
import sys
import mne
import h5py
import dill
import sails
import pprint
import numpy as np
import glmtools as glm
from scipy import io, ndimage, stats
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.append('/Users/andrew/src/qlt')
import qlt

import logging
logger = logging.getLogger('osl')

from glm_config import cfg

outdir = cfg['lemon_analysis_dir']

#%% --------------------------------------------------
# Preprocessing

from lemon_support import (lemon_make_blinks_regressor,
                           lemon_make_task_regressor,
                           lemon_make_bads_regressor,
                           lemon_set_channel_montage,
                           lemon_ica, lemon_check_ica)
config = osl.preprocessing.load_config('lemon_preproc.yml')
pprint.pprint(config)

# 010003 010010 010050 010060 010089
subj_id = '010060'

base = '/Users/andrew/Projects/lemon/EEG_Raw_BIDS_ID/sub-{subj_id}/RSEEG'.format(subj_id=subj_id)
infile = os.path.join(base, 'sub-{subj_id}.vhdr'.format(subj_id=subj_id))
extras = [lemon_set_channel_montage, lemon_ica]
dataset1 = osl.preprocessing.run_proc_chain(infile, config,
                                            extra_funcs=extras,
                                            outname='sub-{subj_id}_proc-full'.format(subj_id=subj_id),
                                            outdir=outdir,
                                            overwrite=True)

fout = os.path.join(outdir, 'sub-{subj_id}_proc-full_flowchart.png'.format(subj_id=subj_id))
osl.preprocessing.plot_preproc_flowchart(config)
plt.savefig(fout, dpi=300, transparent=True)
fout = os.path.join(outdir, 'sub-{subj_id}_proc-full_icatopos.png'.format(subj_id=subj_id))
lemon_check_ica(dataset1, fout)

# ICA CHECK - uncomment to run without ICA denoising in preproc
config['preproc'][8]['lemon_ica']['apply'] = False
dataset2 = osl.preprocessing.run_proc_chain(infile, config,
                                            extra_funcs=extras,
                                            outname='sub-{subj_id}_proc-noica'.format(subj_id=subj_id),
                                            outdir=outdir,
                                            overwrite=True)

fout = os.path.join(outdir, 'sub-{subj_id}_proc-noica_flowchart.png'.format(subj_id=subj_id))
osl.preprocessing.plot_preproc_flowchart(config)
plt.savefig(fout, dpi=300, transparent=True)
fout = os.path.join(outdir, 'sub-{subj_id}_proc-noica_icatopos.png'.format(subj_id=subj_id))
lemon_check_ica(dataset2, fout)

#osl.preprocessing.plot_ica(dataset['ica'], dataset['raw'])

#%% ----------------------------------------------------------
# Loop through datasets

for mode in ['full', 'noica']:

    if mode == 'full':
        dataset = dataset1
    elif mode == 'noica':
        dataset = dataset2

    #%% ----------------------------------------------------------
    # GLM-Prep

    # Make blink regressor
    fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_blink-summary.png'.format(subj_id=subj_id, mode=mode))
    blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(dataset['raw'], figpath=fout)

    veog = dataset['raw'].get_data(picks='ICA-VEOG')[0, :]**2
    thresh = np.percentile(veog, 95)
    veog = veog>thresh

    heog = dataset['raw'].get_data(picks='ICA-HEOG')[0, :]**2
    thresh = np.percentile(heog, 95)
    heog = heog>thresh

    # Make task regressor
    task = lemon_make_task_regressor(dataset)

    # Make bad-segments regressor
    bads = lemon_make_bads_regressor(dataset)

    #bads[blink_vect>0] = 0

    #%% --------------------------------------------------------
    # GLM

    raw = dataset['raw']
    XX = raw.get_data(picks='eeg').T
    XX = stats.zscore(XX, axis=0)

    covs = {'Linear Trend': np.linspace(0, 1, dataset['raw'].n_times),
            'Eyes Open>Closed': task}
    #cons = {'Bad Segments': bads, 'Blinks': blink_vect}
    cons = {'Bad Segments': bads, 'V-EOG': veog, 'H-EOG': heog}
    fs = dataset['raw'].info['sfreq']
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(XX, axis=0,
                                                                    covariates=covs,
                                                                    confounds=cons,
                                                                    nperseg=int(fs*2),
                                                                    fmin=0.1, fmax=100,
                                                                    fs=fs, mode='magnitude',
                                                                    fit_method='glmtools')
    model, design, data = extras
    data.info['dim_labels'] = ['Windows', 'Frequencies', 'Sensors']

    hdfname = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-data.hdf5'.format(subj_id=subj_id, mode=mode))
    if os.path.exists(hdfname):
        print('Overwriting previous results')
        os.remove(hdfname)
    with h5py.File(hdfname) as F:
         model.to_hdf5(F.create_group('model'))
         design.to_hdf5(F.create_group('design'))
         data.to_hdf5(F.create_group('data'))

    fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-design.png'.format(subj_id=subj_id, mode=mode))
    design.plot_summary(show=False, savepath=fout)
    fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-efficiency.png'.format(subj_id=subj_id, mode=mode))
    design.plot_efficiency(show=False, savepath=fout)

    # PERMS

    adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(raw.info, 'eeg')
    ntests = np.prod(data.data.shape[1:])
    ntimes = data.data.shape[1]
    adjacency = mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)

    cft = 3
    tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 1}

    P = []
    run_perms = True
    for icon in range(1, design.num_contrasts):
        fpath = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_perms-con{icon}.pkl'.format(subj_id=subj_id, mode=mode, icon=icon))
        if run_perms:
            p = glm.permutations.MNEClusterPermutation(design, data, icon, 250,
                                                       nprocesses=3,
                                                       metric='tstats',
                                                       tstat_args=tstat_args,
                                                       cluster_forming_threshold=cft,
                                                       adjacency=adjacency)

            with open(fpath, "wb") as dill_file:
                dill.dump(p, dill_file)

            P.append(p)
        else:
            with open(fpath, 'rb') as dill_file:
                P.append(dill.load(dill_file))

    #%% ------------------------------------------------------------


    ll = [['Rec Start', 'Rec End'],
          ['Closed', 'Open'],
          ['Good Seg', 'Bad Seg'],
          ['No Blink', 'Blink']]

    first_row_pos = [0.125, 0.5, 0.13362068965517243, 0.2264705882352941]
    col_heads = ['Mean', 'Linear Trend', 'Rest Condition', 'Bad Segments', 'Blinks']
    col_heads = model.contrast_names

    shade = [0.7, 0.7, 0.7]
    xf = -0.03


    plt.figure(figsize=(12, 6.75))
    for ii in range(5):
        pos = first_row_pos.copy()
        if ii == 0:
            ## Plot Mean Spectrum
            # push left
            pos[0] -= 0.07
            tax = plt.axes(pos)
            ts = model.copes[ii, :, :]
            qlt.plot_joint_spectrum(tax, np.sqrt(ts), raw, xvect=freq_vect, freqs=[1, 9], base=0.5, topo_scale=None)
            qlt.subpanel_label(tax, chr(65), yf=1.6)

            ## Plot Sensor Layout
            pos = first_row_pos.copy()
            pos[1] -= 0.35
            pos[0] -= 0.07
            pax = plt.axes(pos)
            qlt.plot_channel_layout(pax, raw)
            qlt.subpanel_label(pax, chr(65+1), yf=1.04)
        else:
            ## Plot t-stat spectrum
            pos[0] += 0.18*ii
            tax = plt.axes(pos)
            ylabel = 't-stat' if ii == 1 else None
            qlt.plot_sensorspace_clusters(data, P[ii-1], raw, tax, xvect=freq_vect,
                                          ylabel=ylabel, base=0.5, topo_scale=None)
            qlt.subpanel_label(tax, chr(65+(2*ii)), yf=1.6)

            ## Plot model projected spectrum
            pos = first_row_pos.copy()
            pos[1] -= 0.35
            pos[0] += 0.18*ii
            pax = plt.axes(pos)
            proj,llabels = model.project_range(ii, nsteps=2)
            qlt.plot_sensor_data(pax, proj.mean(axis=2).T, raw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=2)
            ylabel = 'Power' if ii == 1 else ''
            qlt.decorate_spectrum(pax, ylabel=ylabel)
            pax.legend(ll[ii-1], frameon=False, fontsize=8)
            qlt.subpanel_label(pax, chr(65+(2*ii)+1), yf=1.04)
        tax.set_title(col_heads[ii])

    fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-summary.png'.format(subj_id=subj_id, mode=mode))
    plt.savefig(fout, dpi=300, transparent=True)


    #%% -------------------------------------------------------------

    ll = [['Rec Start', 'Rec End'],
          ['Closed', 'Open'],
          ['Good Seg', 'Bad Seg'],
          ['No Blink', 'Blink'],
          ['No Blink', 'Blink']]

    col_heads = ['Mean', 'Linear Trend', 'Rest Condition', 'Bad Segments', 'VEOG', 'HEOG']
    #col_heads = model.contrast_names
    refraw = dataset['raw'].copy().pick_types(eeg=True)

    shade = [0.7, 0.7, 0.7]
    xf = -0.03
    plt.figure(figsize=(16, 12))
    plt.subplots_adjust(right=0.957, top=0.95, hspace=0.35, wspace=0.4, bottom=0.04)
    for ii in range(len(model.contrast_names)):
        ind = ii+4 if ii <3 else ii+10
        ax = plt.subplot(6, 3, ind)
        ax2 = plt.subplot(6, 3, ind+3)

        if ii == 0:
            ## Plot Mean Spectrum
            # push left
            ts = model.copes[ii, :, :]
            qlt.plot_joint_spectrum(ax, np.sqrt(ts), refraw, xvect=freq_vect,
                                    freqs=[1, 9, 24], base=0.5, topo_scale=None,
                                    ylabel='Amplitude')
            qlt.subpanel_label(ax, chr(65), yf=1.6)

            ## Plot Sensor Layout
            qlt.plot_channel_layout(ax2, refraw, size=60)
            qlt.subpanel_label(ax2, chr(65+1), yf=1.04)
            pos = list(ax2.get_position().bounds)
            pos[1] -= 0.5
            ax2.set_position(pos)
        else:
            ## Plot t-stat spectrum
            ylabel = 't-stat'
            qlt.plot_sensorspace_clusters(data, P[ii-1], refraw, ax, xvect=freq_vect,
                                          ylabel=ylabel, base=0.5, topo_scale=None)
            qlt.subpanel_label(ax, chr(65+1+ii), yf=1.6)
            ax.set_xlabel('')

            proj,llabels = model.project_range(ii, nsteps=2)
            qlt.plot_sensor_data(ax2, proj.mean(axis=2).T, refraw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=2)
            ylabel = 'Amplitude'
            qlt.decorate_spectrum(ax2, ylabel=ylabel)
            ax2.legend(ll[ii-1], frameon=False, fontsize=8)
            #qlt.subpanel_label(ax2, chr(65+(2*ii)+1), yf=1.04)
        ax.set_title(col_heads[ii])
    fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-summary2.png'.format(subj_id=subj_id, mode=mode))
    plt.savefig(fout, dpi=300, transparent=True)


    #%% -------------------------------------------------------------

    models = glm.fit.run_regressor_selection(design, data)

    plt.figure(figsize=(16, 11))
    plt.subplots_adjust(wspace=0.4, hspace=0.5, top=0.95, bottom=0.05)
    labels = []

    ax = plt.subplot(3, 4, (1,5))
    qlt.plot_sensor_spectrum(ax, models[0].r_square[0, :, :]*100, refraw, freq_vect, base=0.5)
    ax.set_ylabel('R-squared (%)')
    ax.set_title('Full Model')
    labels.append('Full Model')
    qlt.subpanel_label(ax, chr(65))
    pos = list(ax.get_position().bounds)
    pos[1] += 0.1
    pos[3] -= 0.15
    ax.set_position(pos)
    ax.set_ylim(0)

    inds = [2, 3, 4, 6, 7, 8]
    ref = models[0].r_square[0, :, :]
    for ii in range(6):
        ax = plt.subplot(3,4,inds[ii])
        change =  models[1+ii].r_square[0, :, :] * 100
        qlt.plot_sensor_spectrum(ax, change, refraw, freq_vect, base=0.5)
        ax.set_ylabel('R-squared (%)')
        ax.set_title("'{0}' only".format(models[0].regressor_names[ii]))
        labels.append("'{0}' only".format(models[0].regressor_names[ii]))
        qlt.subpanel_label(ax, chr(65+ii+1))
        ax.set_ylim(0)

    ax = plt.subplot(313)
    for ii in range(7):
        x = models[6-ii].r_square.flatten() * 100
        y = np.random.normal(ii+1, 0.05, size=len(x))
        plt.plot(x, y, 'r.', alpha=0.2)
    h = plt.boxplot([m.r_square.flatten() * 100 for m in models[::-1]], vert=False, showfliers=False)
    plt.yticks(np.arange(1,8),labels[::-1])
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    ax.set_xlabel('R-Squared (%)')
    ax.set_ylabel('Model')
    pos = list(ax.get_position().bounds)
    pos[0] += 0.15
    pos[2] -= 0.2
    ax.set_position(pos)
    qlt.subpanel_label(ax, chr(65+6))

    fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-modelselection.png'.format(subj_id=subj_id, mode=mode))
    plt.savefig(fout, dpi=300, transparent=True)

    #%% ------------------------------------------------------------------------


    chan = mne.pick_channels(dataset['raw'].info['ch_names'], ['POz'])[0]

    nperseg = int(fs*2)
    nstep = nperseg/2
    noverlap = nperseg - nstep
    time = np.arange(nperseg/2, dataset['raw'].n_times - nperseg/2 + 1,
                     nperseg - noverlap)/float(dataset['raw'].info['sfreq'])

    vmin = 0
    vmax = 0.0025

    plt.figure(figsize=(16, 10))
    plt.subplots_adjust(right=0.975, top=0.9, hspace=0.4)
    plt.subplot(411)
    plt.pcolormesh(time, freq_vect, data.data[:, :, chan].T, vmin=vmin, vmax=vmax, cmap='hot_r')
    plt.xticks(np.arange(18)*60, np.arange(18))
    plt.ylabel('Frequency (Hz)')
    plt.title('STFT Data')
    plt.colorbar()
    qlt.subpanel_label(plt.gca(), 'A')

    plt.subplot(412)
    plt.pcolormesh(time, np.arange(6), design.design_matrix[:, ::-1].T, cmap='RdBu_r')
    plt.yticks(np.arange(6), model.contrast_names[::-1])
    plt.xticks(np.arange(18)*60, np.arange(18))
    plt.title('Design Matrix')
    plt.colorbar()
    qlt.subpanel_label(plt.gca(), 'B')

    regs = np.arange(3)
    fit = np.dot(design.design_matrix[:, regs], model.betas[regs, :, chan])
    plt.subplot(413)
    plt.xticks(np.arange(18)*60, np.arange(18))
    plt.ylabel('Frequency (Hz)')
    plt.pcolormesh(time, freq_vect, fit.T, vmin=vmin, vmax=vmax, cmap='hot_r')
    plt.title('Mean + Covariate Regressors')
    plt.colorbar()
    qlt.subpanel_label(plt.gca(), 'C')

    regs = np.arange(3)+3
    fit = np.dot(design.design_matrix[:, regs], model.betas[regs, :, chan])
    plt.subplot(414)
    plt.xticks(np.arange(18)*60, np.arange(18))
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (mins)')
    plt.pcolormesh(time, freq_vect, fit.T, vmin=vmin, vmax=0.001, cmap='hot_r')
    plt.title('Confound Regressors Only')
    plt.colorbar()
    qlt.subpanel_label(plt.gca(), 'D')

    fout = os.path.join(outdir, 'sub-{subj_id}_proc-{mode}_glm-singlechanTF.png'.format(subj_id=subj_id, mode=mode))
    plt.savefig(fout, dpi=300, transparent=True)
