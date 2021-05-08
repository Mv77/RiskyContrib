# %%
'''
Example implementations of HARK.ConsumptionSaving.ConsPortfolioModel
'''
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyContribConsumerType, init_risky_contrib
from time import time
from copy import copy

import numpy as np
import seaborn as sns
from tools import pol_funcs_dframe
import os

# %% Base parametrization

# Make the problem infinite horizon
par_infinite_base = init_risky_contrib.copy()
par_infinite_base['cycles']   = 0

# and frictionless to start
par_infinite_base['AdjustPrb'] = 1.0
par_infinite_base['tau'] = 0.0

# Make contribution shares a continuous choice
par_infinite_base['DiscreteShareBool'] = False
par_infinite_base['vFuncBool'] = False

# Temporarily make grids sparser
par_infinite_base.update({
    "aXtraCount": 30,
    "mNrmCount": 30,
    "nNrmCount": 30,
})

# %% A version with the tax
par_infinite_tax = copy(par_infinite_base)
par_infinite_tax['tau'] = 0.1

# %% A version with the Calvo friction
par_infinite_calvo = copy(par_infinite_base)
par_infinite_calvo["AdjustPrb"] = 0.25

# %% Create and solve agents with all the parametrizations

agents = {'Base': RiskyContribConsumerType(**par_infinite_base),
          'Tax': RiskyContribConsumerType(**par_infinite_tax),
          'Calvo': RiskyContribConsumerType(**par_infinite_calvo)}

for agent in agents:
    
    print('Now solving ' + agent)
    t0 = time()
    agents[agent].solve(verbose = True)
    t1 = time()
    print('Solving ' + agent +' took ' + str(t1-t0) + ' seconds.')

# %% Hark plot setup
from HARK.utilities import find_gui, make_figs, determine_platform, test_latex_installation, setup_latex_env_notebook

pf = determine_platform()
try:
    latexExists = test_latex_installation(pf)
except ImportError:  # windows and MacOS requires manual install
    latexExists = False

setup_latex_env_notebook(pf, latexExists)

# Whether to save the figures to Figures_dir
saveFigs = True

# Whether to draw the figures
drawFigs = True

def make(fig, name, target_dir="../../../Figures"):
    fig.savefig(os.path.join(target_dir, "{}.pdf".format(name)))

# %% Plots


t = 0
mNrmGrid = np.linspace(0,50,100)
nNrm_vals = np.array([0.0, 10.0, 20])
Share_vals = np.array([0.0, 0.5])

polfuncs = pol_funcs_dframe(agents, t, mNrmGrid, nNrm_vals, Share_vals)

# General aesthetics

#sns.set(rc={'figure.figsize':(2,2)})
sns.set_style("whitegrid")
sns.set_context("paper",
                font_scale=1.5,
                rc={"lines.linewidth": 2.5})


# Rebalancing fraction
g = sns.FacetGrid(polfuncs[polfuncs.control == "d"], col="n", hue = "model",
                  height=3, aspect=(7/3)*1/3)
g.map(sns.lineplot, "m", "value", alpha=.7, linewidth = 2)
g.add_legend(bbox_to_anchor=[0.5, 0.0], ncol = 3, title = "")
g.set_axis_labels('m', 'Normalized Rebalancing Flow: d')
make(g,'inf_dFunc')

# Share fraction
g = sns.FacetGrid(polfuncs[polfuncs.control == "Share"], col="n", hue = "model",
                  height=3, aspect=(7/3)*1/3)
g.map(sns.lineplot, "m", "value", alpha=.7, linewidth = 2)
g.add_legend(bbox_to_anchor=[0.5, 0.0], ncol = 3, title = "")
g.set_axis_labels(r'$\tilde{m}$', 'Income Deduct. Share')

make(g,'inf_ShareFunc')

# Consumption fraction
g = sns.FacetGrid(polfuncs[polfuncs.control == "c"], col="n", row = "Share", hue = "model",
                  height=3, aspect=(7/3)*1/3)
g.map(sns.lineplot, "m", "value", alpha=.7, linewidth = 2)
g.add_legend(bbox_to_anchor=[0.5, 0.0], ncol = 3, title = "")
g.set_axis_labels(r'$\tilde{m}$', r'Consumption: $c$')

make(g,'inf_cFunc')