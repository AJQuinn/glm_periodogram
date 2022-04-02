import os
import osl
import numpy as np
import mne
import sails
from scipy import io, ndimage
import matplotlib.pyplot as plt

from glm_config import cfg

import logging
logger = logging.getLogger('osl')


def lemon_set_channel_montage(dataset, userargs):
    logger.info('LEMON Stage - load and set channel montage')
    logger.info('userargs: {0}'.format(str(userargs)))

    subj = '010060'
    #base = f'/Users/andrew/Projects/lemon/EEG_Raw_BIDS_ID/sub-{subj}/RSEEG/'
    #base = f'/ohba/pi/knobre/datasets/MBB-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/sub-{subj}/RSEEG/'
    ref_file = os.path.join(cfg['lemon_raw'], f'sub-{subj}', 'RSEEG', f'sub-{subj}.mat')
    X = io.loadmat(ref_file)
    ch_pos = {}
    for ii in range(len(X['Channel'][0])-1):  #final channel is reference
        key = X['Channel'][0][ii][0][0].split('_')[2]
        if key[:2] == 'FP':
            key = 'Fp' + key[2]
        value = X['Channel'][0][ii][3][:, 0]
        value = np.array([value[1], value[0], value[2]])
        ch_pos[key] = value

    dig = mne.channels.make_dig_montage(ch_pos=ch_pos)
    dataset['raw'].set_montage(dig)

    return dataset


def lemon_ica(dataset, userargs, logfile=None):
    logger.info('LEMON Stage - custom EEG ICA function')
    logger.info('userargs: {0}'.format(str(userargs)))

    # NOTE: **userargs doesn't work because 'picks' is in there
    ica = mne.preprocessing.ICA(n_components=userargs['n_components'],
                                max_iter=1000,
                                random_state=42)

    # https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#filtering-to-remove-slow-drifts
    fraw = dataset['raw'].copy().filter(l_freq=1., h_freq=None)

    ica.fit(fraw, picks=userargs['picks'])
    dataset['ica'] = ica

    logger.info('starting EOG autoreject')
    # Find and exclude VEOG
    eog_indices, eog_scores = dataset['ica'].find_bads_eog(dataset['raw'])
    dataset['eog_scores'] = eog_scores
    dataset['ica'].exclude.extend(eog_indices)
    logger.info('Marking {0} ICs as EOG'.format(len(dataset['ica'].exclude)))

    # Find and exclude HEOG
    heog_indices = lemon_find_heog(fraw, ica)
    dataset['ica'].exclude.extend(heog_indices)
    logger.info('Marking {0} ICs as HEOG'.format(len(heog_indices)))

   # Save components as channels in raw object
    src = dataset['ica'].get_sources(fraw).get_data()
    veog = src[eog_indices[np.argmax(eog_scores[eog_indices])], :]
    heog = src[heog_indices[0], :]

    info = mne.create_info(['ICA-VEOG', 'ICA-HEOG'],
                           dataset['raw'].info['sfreq'],
                           ['misc', 'misc'])
    eog_raw = mne.io.RawArray(np.c_[veog, heog].T, info)
    dataset['raw'].add_channels([eog_raw], force_update_info=True)

    # Apply ICA denoising or not
    if ('apply' not in userargs) or (userargs['apply'] is True):
        logger.info('Removing selected components from raw data')
        dataset['ica'].apply(dataset['raw'])
    else:
        logger.info('Components were not removed from raw data')
    return dataset


def camcan_ica(dataset, userargs):
    logger.info('LEMON Stage - custom EEG ICA function')
    logger.info('userargs: {0}'.format(str(userargs)))

    # NOTE: **userargs doesn't work because 'picks' is in there
    ica = mne.preprocessing.ICA(n_components=userargs['n_components'],
                                max_iter=1000,
                                random_state=42)

    # https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#filtering-to-remove-slow-drifts
    fraw = dataset['raw'].copy().filter(l_freq=1., h_freq=None)

    ica.fit(fraw, picks=userargs['picks'])
    dataset['ica'] = ica

    logger.info('starting ICA autoreject')
    # Find and exclude ECG
    ecg_indices, ecg_scores = dataset['ica'].find_bads_ecg(dataset['raw'])
    dataset['ica'].exclude.extend(ecg_indices)
    logger.info('ica.find_bads_ecg marking {0}'.format(ecg_indices))

    # Find and exclude VEOG
    heog_indices, heog_scores = dataset['ica'].find_bads_eog(dataset['raw'], ch_name='EOG061')
    dataset['ica'].exclude.extend(heog_indices)
    logger.info('ica.find_bads_eog arking {0} as H-EOG'.format(ecg_indices))

    tmp = np.load(os.path.join(cfg['code_dir'], 'heog_template.npy'))
    comps = dataset['ica'].get_components()
    C = np.corrcoef(tmp, comps.T)
    heog_template_ind = [np.argmax(C[1:, 0])]

    dataset['ica'].exclude.extend(heog_template_ind)
    logger.info('Template corr marking {0} as H-EOG (corr:{1})'.format(heog_template_ind, C[0, heog_template_ind[0]+1]))

    veog_indices, veog_scores = dataset['ica'].find_bads_eog(dataset['raw'], ch_name='EOG062')
    dataset['ica'].exclude.extend(veog_indices)
    logger.info('ica.find_bads_eog marking {0} as V-EOG'.format(ecg_indices))

    # Get best correlated ICA source and EOGs
    src = dataset['ica'].get_sources(fraw).get_data()
    # Indices are sorted by score so trust that first is best...
    veog = src[veog_indices[0], :]
    heog = src[heog_indices[0], :]
    ecg = src[ecg_indices[0], :]

    info = mne.create_info(['ICA-VEOG', 'ICA-HEOG', 'ICA-ECG'],
                           dataset['raw'].info['sfreq'],
                           ['misc', 'misc', 'misc'])
    eog_raw = mne.io.RawArray(np.c_[veog, heog, ecg].T, info)
    dataset['raw'].add_channels([eog_raw], force_update_info=True)

    # Apply ICA denoising or not
    if ('apply' not in userargs) or (userargs['apply'] is True):
        logger.info('Removing selected components from raw data')
        dataset['ica'].apply(dataset['raw'])
    else:
        logger.info('Components were not removed from raw data')
    return dataset


def camcan_check_ica(dataset, figpath):

    comps = dataset['ica'].get_components()
    plt.figure(figsize=(16, 8))

    chans = mne.pick_types(dataset['raw'].info, 'grad')
    info =  mne.pick_info(dataset['raw'].info, chans)
    for idx, ind in enumerate(dataset['ica'].exclude):
        plt.subplot(2, len(dataset['ica'].exclude), idx+1)
        mne.viz.plot_topomap(comps[chans, ind], info)

    chans = mne.pick_types(dataset['raw'].info, 'mag')
    info =  mne.pick_info(dataset['raw'].info, chans)
    for idx, ind in enumerate(dataset['ica'].exclude):
        plt.subplot(2, len(dataset['ica'].exclude), idx+len(dataset['ica'].exclude)+1)
        mne.viz.plot_topomap(comps[chans, ind], info)

    plt.savefig(figpath, transparent=True, dpi=300)


def lemon_make_task_regressor(dataset):
    ev, ev_id = mne.events_from_annotations(dataset['raw'])
    print('Found {0} events in raw'.format(ev.shape[0]))
    print(ev_id)

    # Correct for cropping first 10 seconds - not sure why this is necessary?!
    ev[:, 0] -= dataset['raw'].first_samp

    task = np.zeros((dataset['raw'].n_times,))
    for ii in range(ev.shape[0]):
        if ev[ii, 2] == ev_id['Stimulus/S200']:
            # EYES OPEN
            task[ev[ii,0]:ev[ii,0]+5000] = 1
        elif ev[ii, 2] == ev_id['Stimulus/S210']:
            # EYES CLOSED
            task[ev[ii,0]:ev[ii,0]+5000] = -1
        elif ev[ii, 2] == 1:
            task[ev[ii,0]] = task[ev[ii,0]-1]

    return task


def lemon_check_ica(dataset, figpath):

    comps = dataset['ica'].get_components()
    plt.figure(figsize=(16, 5))
    for idx, ind in enumerate(dataset['ica'].exclude):
        plt.subplot(1, len(dataset['ica'].exclude), idx+1)
        mne.viz.plot_topomap(comps[:, ind], dataset['raw'].info)

    plt.savefig(figpath, transparent=True, dpi=300)


def lemon_find_heog(raw, ica):

    pos = raw.get_montage().get_positions()['ch_pos']
    coords = np.array([pos[p] for p in pos])

    F7 = np.where([ch == 'F7' for ch in ica.ch_names])[0][0]
    F8 = np.where([ch == 'F8' for ch in ica.ch_names])[0][0]

    comps = ica.get_components()
    heog = np.argmax(np.abs(comps[F7, :] - comps[F8, :]))

    return [heog]


def lemon_make_blinks_regressor(raw, corr_thresh=0.75, figpath=None):
    eog_events = mne.preprocessing.find_eog_events(raw, l_freq=1, h_freq=10)
    logger.info('found {0} blinks'.format(eog_events.shape[0]))

    # Correct for cropping first 10 seconds - not sure why this is necessary?!
    #eog_events[:, 0] -= int(10*raw.info['sfreq'])

    tmin = -0.25
    tmax = 0.5
    epochs = mne.Epochs(raw, eog_events, 998, tmin, tmax, picks='eog')
    ev_eog = epochs.get_data()[:, 0, :]
    C = np.corrcoef(ev_eog.mean(axis=0), ev_eog)[1:,0]
    drops = np.where(C < corr_thresh)[0]
    clean = epochs.copy().drop(drops)
    keeps = np.where(C > corr_thresh)[0]
    dirty = epochs.copy().drop(keeps)

    eog_events = np.delete(eog_events, drops, axis=0)
    logger.info('found {0} clean blinks'.format(eog_events.shape[0]))

    blink_covariate = np.zeros((raw.n_times,))
    blink_covariate[eog_events[:, 0] - raw.first_samp] = 1
    blink_covariate = ndimage.maximum_filter(blink_covariate,
                                             size=raw.info['sfreq']//2)

    if figpath is not None:
        plt.figure(figsize=(16, 10))
        plt.subplot(231)
        plt.plot(epochs.times, epochs.get_data()[:, 0, :].mean(axis=0))
        plt.subplot(234)
        plt.plot(epochs.times, epochs.get_data()[:, 0, :].T)
        plt.subplot(232)
        plt.plot(epochs.times, clean.get_data()[:, 0, :].mean(axis=0))
        plt.subplot(235)
        plt.plot(epochs.times, clean.get_data()[:, 0, :].T)
        plt.subplot(233)
        plt.plot(epochs.times, dirty.get_data()[:, 0, :].mean(axis=0))
        plt.subplot(236)
        plt.plot(epochs.times, dirty.get_data()[:, 0, :].T)
        plt.savefig(figpath, transparent=True, dpi=300)

    return blink_covariate, eog_events.shape[0], clean.average(picks='eog')


def lemon_make_bads_regressor(dataset):
    bads = np.zeros((dataset['raw'].n_times,))
    for an in dataset['raw'].annotations:
        if an['description'].startswith('bad'):
            start = dataset['raw'].time_as_index(an['onset'])[0] - dataset['raw'].first_samp
            duration = int(an['duration'] * dataset['raw'].info['sfreq'])
            bads[start:start+duration] = 1
    return bads

