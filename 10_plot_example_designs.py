import glmtools as glm
import os
import numpy as np
import matplotlib.pyplot as plt

import lemon_plotting

from glm_config import cfg
outdir = cfg['lemon_figures']


#%% ------------------------------------------------
# Top row - simple designs

const = np.ones((128,))
bads = np.zeros((128,))
bads[35:40] = 1
bads[99:108] = 1

X1 = np.vstack((const, bads)).T
regressor_names1 = ['Mean', 'Artefact']
C1 = np.eye(2)
contrast_names1 = ['Mean', 'Artefact']

design1 = glm.design.GLMDesign.initialise_from_matrices(X1, C1,
                                                       regressor_names=regressor_names1,
                                                       contrast_names=contrast_names1)


cond1 = np.repeat([0, 1, 0, 1], 32)
cond2 = np.repeat([1, 0, 1, 0], 32)
bads = np.zeros((128,))
bads[35:40] = 1
bads[99:108] = 1

X2 = np.vstack((cond1, cond2, bads)).T
regressor_names2 = ['Condition1', 'Condition2', 'Artefact']
C2 = np.eye(3)
C2 = np.c_[C2, [1, -1, 0]].T
contrast_names2 = ['Condition1', 'Condition2', 'Artefact', 'Cond1>Cond2']

design2 = glm.design.GLMDesign.initialise_from_matrices(X2, C2,
                                                       regressor_names=regressor_names2,
                                                       contrast_names=contrast_names2)

fig = plt.figure(figsize=(16, 6))
ax = plt.subplot(121)
glm.viz.plot_design_summary(X1, regressor_names1,
                            contrasts=C1,
                            contrast_names=contrast_names1,
                            ax=ax)
lemon_plotting.subpanel_label(ax, 'A')

ax = plt.subplot(122)
glm.viz.plot_design_summary(X2,
                            regressor_names2,
                            contrasts=C2,
                            contrast_names=contrast_names2,
                            ax=ax)
lemon_plotting.subpanel_label(ax, 'B')

fout = os.path.join(outdir, 'glm-spectrum_example-designs-top.png')
plt.savefig(fout, dpi=300, transparent=True)

#%% ------------------------------------------------
# Bottom row - more complex designs

cond1 = np.repeat([0, 1, 0, 1], 32)
cond2 = np.repeat([1, 0, 1, 0], 32)
bads1 = np.zeros((128,))
bads1[35:40] = 1
bads1[99:108] = 1

bads2 = np.sin(2*np.pi*1*np.linspace(0,1,128))
bads2 = bads2 - 0.5
bads2[bads2<0] = 0

cov = np.random.randn(128,) / 2

X3 = np.vstack((cond1, cond2, bads1, bads2, cov)).T
regressor_names3 = ['Condition1', 'Condition2', 'Block Artefact', 'Dynamic Artefact', 'Covariate']
C3 = np.eye(5)
C3 = np.c_[C3, [1, -1, 0, 0 ,0]].T
contrast_names3 = ['Condition1', 'Condition2', 'Block Artefact', 'Dynamic Artefact', 'Covariate', 'Cond1>Cond2']

design3 = glm.design.GLMDesign.initialise_from_matrices(X3, C3,
                                                       regressor_names=regressor_names3,
                                                       contrast_names=contrast_names3)

fig = plt.figure(figsize=(16, 6))

ax = plt.subplot(111)
glm.viz.plot_design_summary(X3, regressor_names3, contrasts=C3, contrast_names=contrast_names3, ax=ax)
lemon_plotting.subpanel_label(ax, 'C')


fout = os.path.join(outdir, 'glm-spectrum_example-designs-bottom.png')
plt.savefig(fout, dpi=300, transparent=True)
