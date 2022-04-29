import numpy as np
import dill
import os
import sys
from scipy import stats
import glmtools as glm
from anamnesis import obj_from_hdf5file
from copy import deepcopy

from glm_config import cfg

sys.path.append('/Users/andrew/src/qlt')
import qlt

outdir = '/Users/andrew/Projects/glm/glm_psd/analysis'
figbase = '/Users/andrew/Projects/glm/glm_psd/figures/'

#sens = np.setdiff1d(np.arange(306), np.arange(2,306,3))  # Grads

#%% ---------------------------------
fstart = 2
fstop = 150

print('Loading CamCAN')
camcan_data = obj_from_hdf5file(os.path.join(outdir, 'camcan_meg_sensorglm_groupdata.hdf5'), 'data')
#camcan_data.data = camcan_data.data[:, :, :, fstart:fstop]
camcan_data.info['blinkpm'] = np.array(camcan_data.info['num_blinks']) / np.array(camcan_data.info['scan_duration'])

camcan_data.data = camcan_data.data
camcan_data.data[np.isnan(camcan_data.data)] = 0

drops = [h == None for  h in camcan_data.info['height']]
camcan_data = camcan_data.drop(drops)
drops = [h == None for  h in camcan_data.info['tiv_cubicmm']]
camcan_data = camcan_data.drop(drops)

camcan_data.info['gender_code'] = camcan_data.info['gender_code'].astype(int)
mn = np.min(camcan_data.info['tiv_cubicmm'])
camcan_data.info['tiv_cubicmm2'] = np.power(camcan_data.info['tiv_cubicmm']-mn,2).astype(int)
camcan_data.info['tiv_cubicmm'] = camcan_data.info['tiv_cubicmm'].astype(int)

#drops = np.where(np.isnan(camcan_data.info['age']))[0]
#camcan_data = camcan_data.drop(drops)
#drops = np.where(np.isnan(camcan_data.info['tiv']))[0]
#camcan_data = camcan_data.drop(drops)
#drops = np.where(camcan_data.info['task'] == 1)[0]
#camcan_data = camcan_data.drop(drops)

#camcan_data = camcan_data.drop(np.arange(100, len(camcan_data.info['subj'])))

#camcan_data.data = np.swapaxes(camcan_data.data[:, :, sens, :], 2, 3)

sample_rate = 250
nperseg = 500
freq_vect = np.linspace(0, sample_rate/2, nperseg//2)[:200]

fl_cons = ['Mean', 'HeartRate', 'LinearTrend', 'V-EOG', 'BadSegs']

#%% ---------------------------------

import mne
raw = mne.io.read_raw_fif('/Users/andrew/Projects/COVID/data/cmore_data/cmo001027_rest_tsss.fif', preload=False)
raw.pick_types(meg='grad')

adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(raw.info, 'grad')
ntests = np.prod(camcan_data.data.shape[2:])
ntimes = camcan_data.data.shape[2]
adjacency = mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)

cft = 4
tstat_args = {'hat_factor': 5e-3, 'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 1}

# Mean data for permutation
fl_mean_data = deepcopy(camcan_data)
fl_mean_data.data = camcan_data.data[:, 0, : ,:]

#%% --------------------------------

DC = glm.design.DesignConfig()
DC.add_regressor(name='Mean', rtype='Constant', datainfo='gender_code', codes=2)
DC.add_regressor(name='Female>Male', rtype='Parametric', datainfo='gender_code', preproc='z')
DC.add_regressor(name='HeartRate', rtype='Parametric', datainfo='ecg_bpm', preproc='z')

DC.add_simple_contrasts()

des1 = DC.design_from_datainfo(camcan_data.info)
camcan_model1 = glm.fit.OLSModel(des1, camcan_data)

fout = os.path.join(outdir, 'camcan-group_glm-simple_design.png')
des1.plot_summary(savepath=fout, show=False)
fout = os.path.join(outdir, 'camcan-group_glm-simple_efficiency.png')
des1.plot_efficiency(savepath=fout, show=False)

icon = 1
run_perms = True
if run_perms:
    P1 = glm.permutations.MNEClusterPermutation(des1, fl_mean_data, icon, 100,

                                                cluster_forming_threshold=cft,
                                                tstat_args=tstat_args,
                                                adjacency=adjacency)
    with open(os.path.join(outdir, 'camcan-group_glm-simple_perms-con{0}.pkl'.format(icon)), "wb") as dill_file:
        dill.dump(P1, dill_file)

else:
    dill_file = os.path.join(outdir, 'camcan-group_glm-simple_perms-con{0}.pkl'.format(icon))
    P1 = dill.load(open(dill_file, 'rb'))

#%% --------------------------------

DC = glm.design.DesignConfig()
DC.add_regressor(name='Mean', rtype='Constant', datainfo='gender_code', codes=2)
DC.add_regressor(name='Female>Male', rtype='Parametric', datainfo='gender_code', preproc='z')
DC.add_regressor(name='HeadSize', rtype='Parametric', datainfo='tiv_cubicmm', preproc='z')

DC.add_simple_contrasts()

des2 = DC.design_from_datainfo(camcan_data.info)
camcan_model2 = glm.fit.OLSModel(des2, camcan_data)

fout = os.path.join(outdir, 'camcan-group_glm-headcov_design.png')
des2.plot_summary(savepath=fout, show=False)
fout = os.path.join(outdir, 'camcan-group_glm-headcov_efficiency.png')
des2.plot_efficiency(savepath=fout, show=False)

run_perms = True
if run_perms:
    icon = 1
    P2 = glm.permutations.MNEClusterPermutation(des2, fl_mean_data, icon, 100,
                                                nprocesses=5,
                                                metric='tstats',
                                                cluster_forming_threshold=cft,
                                                tstat_args=tstat_args,
                                                adjacency=adjacency)
    with open(os.path.join(outdir, 'camcan-group_glm-headcov_perms-con{0}.pkl'.format(icon)), "wb") as dill_file:
        dill.dump(P2, dill_file)

    icon = 2
    P3 = glm.permutations.MNEClusterPermutation(des2, fl_mean_data, icon, 100,
                                                nprocesses=5,
                                                metric='tstats',
                                                cluster_forming_threshold=cft,
                                                tstat_args=tstat_args,
                                                adjacency=adjacency)
    with open(os.path.join(outdir, 'camcan-group_glm-headcov_perms-con{0}.pkl'.format(icon)), "wb") as dill_file:
        dill.dump(P3, dill_file)

else:
    icon = 1
    dill_file = os.path.join(outdir, 'camcan-group_glm-headcov_perms-con{0}.pkl'.format(icon))
    P2 = dill.load(open(dill_file, 'rb'))
    icon = 2
    dill_file = os.path.join(outdir, 'camcan-group_glm-headcov_perms-con{0}.pkl'.format(icon))
    P3 = dill.load(open(dill_file, 'rb'))

#%% --------------------------------


plt.figure(figsize=(16, 9))

ax = plt.axes([0.05, 0.45, 0.2, 0.3])
qlt.plot_joint_spectrum(ax, camcan_model1.copes[0, 0, :, :],
                        raw, freq_vect, base=0.5,
                        freqs=[1, 10, 22], topo_scale=None)
qlt.subpanel_label(ax, chr(65), yf=1.1)

ax = plt.axes([0.075, 0.05, 0.15, 0.225])
qlt.plot_channel_layout(ax, raw, size=50)
qlt.subpanel_label(ax, chr(65+1), yf=1.1)


ax = plt.axes([0.3, 0.475, 0.18, 0.27])
ax.set_ylim(-13, 13)
qlt.plot_sensorspace_clusters(fl_mean_data, P1, raw, ax,
                              base=0.5, title='Female>Male Simple Model',
                              ylabel='t-stat', thresh=95, xvect=freq_vect)

qlt.subpanel_label(ax, chr(65+2), yf=1.1)

ax = plt.axes([0.55, 0.475, 0.18, 0.27])
qlt.plot_sensorspace_clusters(fl_mean_data, P2, raw, ax,
                              base=0.5, title='Female>Male Headsize Model',
                              ylabel='t-stat', thresh=95, xvect=freq_vect)
ax.set_ylim(-13, 13)

qlt.subpanel_label(ax, chr(65+4), yf=1.1)

ax = plt.axes([0.8, 0.475, 0.18, 0.27])
qlt.plot_sensorspace_clusters(fl_mean_data, P3, raw, ax,
                              base=0.5, title='Headsize Covariate',
                              ylabel='t-stat', thresh=95, xvect=freq_vect)

qlt.subpanel_label(ax, chr(65+6), yf=1.1)

ch_ind = 156
ax = plt.axes([0.3, 0.1, 0.18, 0.27])
proj, ll = camcan_model1.project_range(1)
qlt.plot_sensor_data(ax, proj[:, 0, : ,ch_ind].T, raw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=2)
qlt.decorate_spectrum(ax)
plt.legend(['Female', 'Male'], frameon=False)
qlt.subpanel_label(ax, chr(65+3), yf=1.1)


ax = plt.axes([0.55, 0.1, 0.18, 0.27])
proj, ll = camcan_model2.project_range(1)
qlt.plot_sensor_data(ax, proj[:, 0, : ,ch_ind].T, raw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=2)
qlt.decorate_spectrum(ax)
plt.legend(['Female', 'Male'], frameon=False)
qlt.subpanel_label(ax, chr(65+5), yf=1.1)

ax = plt.axes([0.8, 0.1, 0.18, 0.27])
proj, ll = camcan_model2.project_range(2, nsteps=5)
qlt.plot_sensor_data(ax, proj[:, 0, :, ch_ind].T, raw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=1)
#ax.set_ylim(0, 0.2)
#ax.set_yticks(np.linspace(0,0.2,5))
qlt.decorate_spectrum(ax)
lll = ['Smallest', 'Small', 'Average', 'Large', 'Largest']
plt.legend(lll, frameon=False, title='HeadSize')
qlt.subpanel_label(ax, chr(65+7), yf=1.1)

fout = os.path.join(outdir, 'camcan-group_glm-headcov_results.png')
plt.savefig(fout, transparent=True, dpi=300)
