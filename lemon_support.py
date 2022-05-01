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


def lemon_create_heog(dataset, userargs):
    logger.info('LEMON Stage - Create HEOG from F7 and F8')
    logger.info('userargs: {0}'.format(str(userargs)))

    F7 = dataset['raw'].get_data(picks='F7')
    F8 = dataset['raw'].get_data(picks='F8')

    heog = F7-F8

    info = mne.create_info(['HEOG'],
                           dataset['raw'].info['sfreq'],
                           ['eog'])
    eog_raw = mne.io.RawArray(heog, info)
    dataset['raw'].add_channels([eog_raw], force_update_info=True)

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
    #eog_indices, eog_scores = dataset['ica'].find_bads_eog(dataset['raw'])
    veog_indices, eog_scores =  dataset['ica'].find_bads_eog(dataset['raw'], 'VEOG')
    dataset['veog_scores'] = eog_scores
    dataset['ica'].exclude.extend(veog_indices)
    logger.info('Marking {0} ICs as EOG {1}'.format(len(dataset['ica'].exclude),
                                                    veog_indices))

    # Find and exclude HEOG
    #heog_indices = lemon_find_heog(fraw, ica)
    heog_indices, eog_scores =  dataset['ica'].find_bads_eog(dataset['raw'], 'HEOG')
    dataset['heog_scores'] = eog_scores
    dataset['ica'].exclude.extend(heog_indices)
    logger.info('Marking {0} ICs as HEOG {1}'.format(len(heog_indices),
                                                     heog_indices))

   # Save components as channels in raw object
    src = dataset['ica'].get_sources(fraw).get_data()
    veog = src[veog_indices[0], :]
    heog = src[heog_indices[0], :]

    ica.labels_['top'] = [veog_indices[0], heog_indices[0]]

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
    logger.info('CamCAN Stage - custom MEG ICA function')
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
    # https://mne.tools/0.18/auto_tutorials/preprocessing/plot_artifacts_correction_ica.html

    # Find and exclude ECG
    ecg_epochs = mne.preprocessing.create_ecg_epochs(fraw, tmin=-.5, tmax=.5)
    ecg_indices, ecg_scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
    if len(ecg_indices) > 0:
        dataset['ica'].exclude.extend(ecg_indices)
    logger.info('ica.find_bads_ecg marking {0}'.format(ecg_indices))

    # Find and exclude HEOG - pretty much the same as VEOG in MEG
    heog_indices, heog_scores = dataset['ica'].find_bads_eog(fraw, ch_name='EOG061')
    if len(heog_indices) > 0:
    	dataset['ica'].exclude.extend(heog_indices)
    logger.info('ica.find_bads_eog marking {0} as H-EOG'.format(heog_indices))

    # Find and exclude VEOG
    eog_epochs = mne.preprocessing.create_eog_epochs(fraw)  # get single EOG trials
    veog_indices, scores = ica.find_bads_eog(eog_epochs)  # find via correlation
    if len(veog_indices) > 0:
        dataset['ica'].exclude.extend(veog_indices)
    logger.info('ica.find_bads_eog marking {0} as V-EOG'.format(veog_indices))

    # Get best correlated ICA source and EOGs
    src = dataset['ica'].get_sources(fraw).get_data()

    # Indices are sorted by score so trust that first is best...
    if len(veog_indices) > 0:
        veog = src[veog_indices[0], :]
    else:
        veog = src[heog_indices[0], :]
    ecg = src[ecg_indices[0], :]

    info = mne.create_info(['ICA-VEOG', 'ICA-ECG'],
                           dataset['raw'].info['sfreq'],
                           ['misc', 'misc'])
    eog_raw = mne.io.RawArray(np.c_[veog, ecg].T, info)
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


def find_eog_events(raw, event_id=998):
    eog = raw.copy().filter(l_freq=1, h_freq=10, picks='eog').get_data(picks='VEOG')
    eog = eog[0, :]
    # 10 seconds hopefully long enough to avoid rejecting real blinks - only
    # want to catch HUGE artefacts here.
    bads = sails.utils.detect_artefacts(eog, axis=0, reject_mode='segments', segment_len=2500)
    eog[bads] = np.median(eog)

    if np.abs(np.max(eog)) > np.abs(np.min(eog)):
        eog_events, _ = mne.preprocessing.eog.peak_finder(eog,
                                                          None, extrema=1)
    else:
        eog_events, _ = mne.preprocessing.eog.peak_finder(eog,
                                                          None, extrema=-1)

    n_events = len(eog_events)
    #logger.info(f'Number of EOG events detected: {n_events}')
    eog_events = np.array([eog_events + raw.first_samp,
                           np.zeros(n_events, int),
                           event_id * np.ones(n_events, int)]).T

    return eog_events


def make_eog_regressors(raw):

    heog = raw.copy().filter(l_freq=1, h_freq=10, picks='ICA-HEOG').get_data(picks='ICA-HEOG')
    heog = heog[0, :]

    heog = stats.zscore(heog).reshape(-1,1)
    bads = sails.utils.detect_artefacts(heog, axis=0, reject_mode='segments', segment_len=2500)
    heog[bads] = np.median(heog)

    gmm = GaussianMixture(2, n_init=5).fit(heog)
    heog = gmm.predict(heog) == np.argmax(gmm.means_)

    veog = raw.copy().filter(l_freq=1, h_freq=10, picks='ICA-VEOG').get_data(picks='ICA-VEOG')
    veog = veog[0, :]

    veog = stats.zscore(veog).reshape(-1,1)
    bads = sails.utils.detect_artefacts(veog, axis=0, reject_mode='segments', segment_len=2500)
    veog[bads] = np.median(veog)

    gmm = GaussianMixture(2, n_init=5).fit(veog)
    veog = gmm.predict(veog) == np.argmax(gmm.means_)


def lemon_make_blinks_regressor(raw, corr_thresh=0.75, figpath=None):
    #eog_events = mne.preprocessing.find_eog_events(raw, l_freq=1, h_freq=10)
    eog_events = find_eog_events(raw)
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


def quick_plot_eogs(raw, ica, figpath=None):

    inds = np.arange(250*45, 250*300)

    plt.figure(figsize=(24, 9))
    veog = raw.get_data(picks='VEOG')[0, :]
    ica_veog = raw.get_data(picks='ICA-VEOG')[0, :]
    plt.axes([0.05, 0.55, 0.125, 0.4])
    comp = ica.get_components()[:, ica.labels_['top'][0]]
    mne.viz.plot_topomap(comp, ica.info)
    plt.axes([0.2, 0.55, 0.475, 0.4])
    plt.plot(stats.zscore(veog[inds]))
    plt.plot(stats.zscore(ica_veog[inds])-5)
    plt.legend(['VEOGs', 'ICA-VEOG'], frameon=False)
    plt.xlim(0, 250*180)
    plt.axes([0.725, 0.55, 0.25, 0.4])
    plt.plot(veog, ica_veog, '.k')
    veog = raw.get_data(picks='VEOG', reject_by_annotation='omit')[0, :]
    ica_veog = raw.get_data(picks='ICA-VEOG', reject_by_annotation='omit')[0, :]
    plt.plot(veog, ica_veog, '.r')
    plt.xlabel('VEOG'); plt.ylabel('ICA-VEOG')
    plt.plot(veog, ica_veog, '.r')
    plt.legend(['Samples', 'Clean Samples'], frameon=False)
    plt.title('Correlation : r = {0}'.format(np.corrcoef(veog, ica_veog)[0,  1]))

    heog = raw.get_data(picks='HEOG')[0, :]
    ica_heog = raw.get_data(picks='ICA-HEOG')[0, :]
    plt.axes([0.05, 0.05, 0.125, 0.4])
    comp = ica.get_components()[:, ica.labels_['top'][1]]
    mne.viz.plot_topomap(comp, ica.info)
    plt.axes([0.2, 0.05, 0.475, 0.4])
    plt.plot(stats.zscore(heog[inds]))
    plt.plot(stats.zscore(ica_heog[inds])-5)
    plt.legend(['HEOGs', 'ICA-HEOG'], frameon=False)
    plt.xlim(0, 250*180)
    plt.axes([0.725, 0.05, 0.25, 0.4])
    plt.plot(heog, ica_heog, '.k')
    heog = raw.get_data(picks='HEOG', reject_by_annotation='omit')[0, :]
    ica_heog = raw.get_data(picks='ICA-HEOG', reject_by_annotation='omit')[0, :]
    plt.plot(heog, ica_heog, '.r')
    plt.legend(['Samples', 'Clean Samples'], frameon=False)
    plt.xlabel('HEOG'); plt.ylabel('ICA-HEOG')
    plt.title('Correlation : r = {0}'.format(np.corrcoef(heog, ica_heog)[0,  1]))

    plt.savefig(figpath, transparent=False, dpi=300)
