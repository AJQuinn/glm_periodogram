import numpy as np
import dill
import os
import sys
from scipy import stats
import glmtools as glm
from anamnesis import obj_from_hdf5file
from copy import deepcopy

sys.path.append('/Users/andrew/src/qlt')
import qlt

outdir = '/Users/andrew/Projects/glm/glm_psd/analysis'
figbase = '/Users/andrew/Projects/glm/glm_psd/figures/'

sens = np.setdiff1d(np.arange(306), np.arange(2,306,3))  # Grads

#%% ---------------------------------
fstart = 2
fstop = 150

print('Loading CamCAN')
base = '/Users/andrew/Projects/meguk_spectra/'
camcan_data = obj_from_hdf5file(base + 'camcan_sensorglm_groupdata.hdf5', 'data')
camcan_data.data = camcan_data.data[:, :, :, fstart:fstop]
camcan_data.info['beatpm'] = camcan_data.info['bpm']
camcan_data.info['blinkpm'] = np.array(camcan_data.info['nblinks']) / np.array(camcan_data.info['scandur'])

camcan_data.data = camcan_data.data
camcan_data.data[np.isnan(camcan_data.data)] = 0

drops = np.where(np.isnan(camcan_data.info['age']))[0]
camcan_data = camcan_data.drop(drops)
drops = np.where(np.isnan(camcan_data.info['tiv']))[0]
camcan_data = camcan_data.drop(drops)
drops = np.where(camcan_data.info['task'] == 1)[0]
camcan_data = camcan_data.drop(drops)

camcan_data = camcan_data.drop(np.arange(100, len(camcan_data.info['subj'])))

camcan_data.data = np.swapaxes(camcan_data.data[:, :, sens, :], 2, 3)

freq_vect = np.linspace(0, 200, 512)[fstart:fstop]

#%% ---------------------------------

import mne
raw = mne.io.read_raw_fif('/Users/andrew/Projects/COVID/data/cmore_data/cmo001027_rest_tsss.fif', preload=False)
raw.pick_types(meg='grad')

adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(raw.info, 'grad')
ntests = np.prod(camcan_data.data.shape[2:])
ntimes = camcan_data.data.shape[2]
adjacency = mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)

cft = 2.3
tstat_args = {'sigma_hat': 'auto'}

# Mean data for permutation
fl_mean_data = deepcopy(camcan_data)
fl_mean_data.data = camcan_data.data[:, 0, : ,:]

#%% --------------------------------

DC = glm.design.DesignConfig()
DC.add_regressor(name='Female', rtype='Categorical', datainfo='sex', codes=1)
DC.add_regressor(name='Male', rtype='Categorical', datainfo='sex', codes=0)

DC.add_simple_contrasts()
DC.add_contrast(name='Mean', values={'Female': 0.5, 'Male': 0.5})
DC.add_contrast(name='Female>Male', values={'Female': 1, 'Male': -1})

des1 = DC.design_from_datainfo(camcan_data.info)
camcan_model1 = glm.fit.OLSModel(des1, camcan_data)

fout = os.path.join(outdir, 'camcan-group_glm-simple_design.png')
des1.plot_summary(savepath=fout, show=False)
fout = os.path.join(outdir, 'camcan-group_glm-simple_efficiency.png')
des1.plot_efficiency(savepath=fout, show=False)

icon = 3
run_perms = False
if run_perms:
    P1 = glm.permutations.MNEClusterPermutation(des1, fl_mean_data, icon, 200,
                                                nprocesses=3,
                                                metric='tstats',
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
DC.add_regressor(name='Female', rtype='Categorical', datainfo='sex', codes=1)
DC.add_regressor(name='Male', rtype='Categorical', datainfo='sex', codes=0)
DC.add_regressor(name='HeadSize', rtype='Parametric', datainfo='tiv', preproc='z')

DC.add_simple_contrasts()
DC.add_contrast(name='Mean', values={'Female': 0.5, 'Male': 0.5})
DC.add_contrast(name='Female>Male', values={'Female': 1, 'Male': -1})

des2 = DC.design_from_datainfo(camcan_data.info)
camcan_model2 = glm.fit.OLSModel(des2, camcan_data)

fout = os.path.join(outdir, 'camcan-group_glm-headcov_design.png')
des2.plot_summary(savepath=fout, show=False)
fout = os.path.join(outdir, 'camcan-group_glm-headcov_efficiency.png')
des2.plot_efficiency(savepath=fout, show=False)

icon = 3
run_perms = True
if run_perms:
    icon = 4
    P2 = glm.permutations.MNEClusterPermutation(des2, fl_mean_data, icon, 200,
                                                nprocesses=3,
                                                metric='tstats',
                                                cluster_forming_threshold=cft,
                                                tstat_args=tstat_args,
                                                adjacency=adjacency)
    with open(os.path.join(outdir, 'camcan-group_glm-headcov_perms-con{0}.pkl'.format(icon)), "wb") as dill_file:
        dill.dump(P2, dill_file)

    icon = 2
    P3 = glm.permutations.MNEClusterPermutation(des2, fl_mean_data, icon, 200,
                                                nprocesses=3,
                                                metric='tstats',
                                                cluster_forming_threshold=cft,
                                                tstat_args=tstat_args,
                                                adjacency=adjacency)
    with open(os.path.join(outdir, 'camcan-group_glm-headcov_perms-con{0}.pkl'.format(icon)), "wb") as dill_file:
        dill.dump(P3, dill_file)

else:
    icon = 3
    dill_file = os.path.join(outdir, 'camcan-group_glm-headcov_perms-con{0}.pkl'.format(icon))
    P2 = dill.load(open(dill_file, 'rb'))
    icon = 4
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
qlt.plot_sensorspace_clusters(fl_mean_data, P1, raw, ax,
                              base=0.5, title='Female>Male Simple Model',
                              ylabel='t-stat', thresh=99, xvect=freq_vect)
ax.set_ylim(-4.5, 4.5)

qlt.subpanel_label(ax, chr(65+2), yf=1.1)

ax = plt.axes([0.55, 0.475, 0.18, 0.27])
qlt.plot_sensorspace_clusters(fl_mean_data, P2, raw, ax,
                              base=0.5, title='Female>Male Headsize Model',
                              ylabel='t-stat', thresh=99, xvect=freq_vect)
ax.set_ylim(-4.5, 4.5)

qlt.subpanel_label(ax, chr(65+4), yf=1.1)

ax = plt.axes([0.8, 0.475, 0.18, 0.27])
qlt.plot_sensorspace_clusters(fl_mean_data, P3, raw, ax,
                              base=0.5, title='Headsize Covariate',
                              ylabel='t-stat', thresh=99, xvect=freq_vect)

qlt.subpanel_label(ax, chr(65+6), yf=1.1)

ch_ind = 156
ax = plt.axes([0.3, 0.1, 0.18, 0.27])
qlt.plot_sensor_data(ax, camcan_model1.copes[0, 0, : ,ch_ind], raw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=2)
qlt.plot_sensor_data(ax, camcan_model1.copes[1, 0, : ,ch_ind], raw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=2)
ax.set_ylim(0, 0.2)
ax.set_yticks(np.linspace(0,0.2,5))
qlt.decorate_spectrum(ax)
plt.legend(['Female', 'Male'], frameon=False)
qlt.subpanel_label(ax, chr(65+3), yf=1.1)


ax = plt.axes([0.55, 0.1, 0.18, 0.27])
qlt.plot_sensor_data(ax, camcan_model2.copes[0, 0, : ,ch_ind], raw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=2)
qlt.plot_sensor_data(ax, camcan_model2.copes[1, 0, : ,ch_ind], raw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=2)
ax.set_ylim(0, 0.2)
ax.set_yticks(np.linspace(0,0.2,5))
qlt.decorate_spectrum(ax)
plt.legend(['Female', 'Male'], frameon=False)
qlt.subpanel_label(ax, chr(65+5), yf=1.1)

ax = plt.axes([0.8, 0.1, 0.18, 0.27])
proj, ll = camcan_model2.project_range(2, nsteps=5)
qlt.plot_sensor_data(ax, proj[:, 0, :, ch_ind].T, raw, xvect=freq_vect, base=0.5, sensor_cols=False, lw=2)
ax.set_ylim(0, 0.2)
ax.set_yticks(np.linspace(0,0.2,5))
qlt.decorate_spectrum(ax)
plt.legend(ll, frameon=False, title='HeadSize')
qlt.subpanel_label(ax, chr(65+7), yf=1.1)

fout = os.path.join(outdir, 'camcan-group_glm-headcov_results.png')
plt.savefig(fout, transparent=True, dpi=300)
