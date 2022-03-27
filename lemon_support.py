import os
import osl
import numpy as np
import mne
import sails
from scipy import io, ndimage
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('osl')



def lemon_set_channel_montage(dataset, userargs):
    logger.info('LEMON Stage - load and set channel montage')
    logger.info('userargs: {0}'.format(str(userargs)))

    subj = '010060'
    base = f'/Users/andrew/Projects/lemon/EEG_Raw_BIDS_ID/sub-{subj}/RSEEG/'
    X = io.loadmat(base+f'sub-{subj}.mat')
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
    eog_indices, eog_scores = dataset['ica'].find_bads_eog(dataset['raw'])
    dataset['eog_scores'] = eog_scores

    dataset['ica'].exclude.extend(eog_indices)
    logger.info('Marking {0} ICs as EOG'.format(len(dataset['ica'].exclude)))

    heog_indices = lemon_find_heog(fraw, ica)

    dataset['ica'].exclude.extend(heog_indices)
    logger.info('Marking {0} ICs as HEOG'.format(len(heog_indices)))

    src = dataset['ica'].get_sources(fraw).get_data()
    dataset['veog'] = src[eog_indices[np.argmax(eog_scores[eog_indices])], :]
    dataset['heog'] = src[heog_indices[0], :]

    if ('apply' not in userargs) or (userargs['apply'] is True):
        logger.info('Removing selected components from raw data')
        dataset['ica'].apply(dataset['raw'])
    else:
        logger.info('Components were not removed from raw data')
    return dataset


def lemon_make_task_regressor(dataset):
    ev, ev_id = mne.events_from_annotations(dataset['raw'])

    # Correct for cropping first 10 seconds - not sure why this is necessary?!
    ev[:, 0] -= int(10*dataset['raw'].info['sfreq'])

    task = np.zeros((dataset['raw'].n_times,))
    for ii in range(ev.shape[0]):
        if ev[ii, 2] == 200:
            task[ev[ii,0]:ev[ii,0]+5000] = 1
        elif ev[ii, 2] == 210:
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
            start = dataset['raw'].time_as_index(an['onset'])[0]
            duration = int(an['duration'] * dataset['raw'].info['sfreq'])
            bads[start:start+duration] = 1
    return bads

