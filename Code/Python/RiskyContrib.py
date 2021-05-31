# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.9
#   latex_envs:
#     LaTeX_envs_menu_present: true
#     autoclose: false
#     autocomplete: true
#     bibliofile: biblio.bib
#     cite_by: apalike
#     current_citInitial: 1
#     eqLabelWithNumbers: true
#     eqNumInitial: 1
#     hotkeys:
#       equation: Ctrl-E
#       itemize: Ctrl-I
#     labels_anchors: false
#     latex_user_defs: false
#     report_style_numbering: false
#     user_envs_cfg: false
# ---

# %% [markdown]
# # A Two-Asset Savings Model with an Income-Contribution Scheme
#
# ## Mateo Velásquez-Giraldo
#
# This notebook demonstrates the use of the `RiskyContrib` agent type
# of the [Hark Toolkit](https://econ-ark.org/toolkit). The model represents an agent who can
# save using two different assets---one risky and the other risk-free---to insure
# against fluctuations in his income, but faces frictions to transferring funds between
# assets. The flexibility of its implementation and its inclusion in the HARK
# toolkit will allow users to adapt the model to realistic life-cycle calibrations, and
# also to embedded it in heterogeneous-agents macroeconomic models.

# %%
# Preamble

from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    RiskyContribConsumerType,
    init_risky_contrib_lifecycle,
)
from time import time
from copy import copy

import numpy as np
import seaborn as sns
import pandas as pd
from Simulations.tools import pol_funcs_dframe, age_profiles
import os

# This is a jupytext paired notebook that autogenerates a .py file
# which can be executed from a terminal command line
# But a terminal does not permit inline figures, so we need to test jupyter vs terminal
# Google "how can I check if code is executed in the ipython notebook"
from IPython import get_ipython # In case it was run from python instead of ipython

# If the ipython process contains 'terminal' assume not in a notebook
def in_ipynb():
    try:
        if 'terminal' in str(type(get_ipython())):
            return False
        else:
            return True
    except NameError:
        return False
    
# Determine whether to make the figures inline (for spyder or jupyter)
# vs whatever is the automatic setting that will apply if run from the terminal
if in_ipynb():
    # %matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic('matplotlib', 'inline')
else:
    get_ipython().run_line_magic('matplotlib', 'auto')

# %% [markdown]
# ## Model Description

# %% [markdown]
# ## Parametrizations

# %%
# %% Base parametrization

# Make the problem life-cycle
par_LC_base = init_risky_contrib_lifecycle.copy()

# Turn off aggregate growth
par_LC_base['PermGroFacAgg'] = 1.0

# and frictionless to start
par_LC_base["AdjustPrb"] = 1.0
par_LC_base["tau"] = 0.0

# Make contribution shares a continuous choice
par_LC_base["DiscreteShareBool"] = False
par_LC_base["vFuncBool"] = False

# Temporarily make grids sparser
par_LC_base.update(
    {"aXtraCount": 30, "mNrmCount": 30, "nNrmCount": 30,
     "mNrmMax": 500, "nNrmMax":500}
)

# %% A version with the tax
par_LC_tax = copy(par_LC_base)
par_LC_tax["tau"] = 0.1

# %% A version with the Calvo friction
par_LC_calvo = copy(par_LC_base)
par_LC_calvo["AdjustPrb"] = 0.25

# %% A calibration with a probability and tax that change at retirement
par_LC_retirement = copy(par_LC_base)
par_LC_retirement["AdjustPrb"] = [1.0] + [0.0]*39 + [1.0]*25
par_LC_retirement["tau"] = [0.0]*41 + [0.0]*24
par_LC_retirement["UnempPrb"] = 0.0

# %% [markdown]
# # Solution and policy functions

# %%
# %% Create and solve agents with all the parametrizations
agents = {
    "Base": RiskyContribConsumerType(**par_LC_base),
    "Tax": RiskyContribConsumerType(**par_LC_tax),
    "Calvo": RiskyContribConsumerType(**par_LC_calvo),
    "Retirement": RiskyContribConsumerType(**par_LC_retirement),
}

for agent in agents:

    print("Now solving " + agent)
    t0 = time()
    agents[agent].solve(verbose=True)
    t1 = time()
    print("Solving " + agent + " took " + str(t1 - t0) + " seconds.")


# %% [markdown]
# ## Simulation and average life-cycle profiles

# %%
# %% Solve and simulate
n_agents = 10
t_sim    = 500
profiles = []
for agent in agents:
    agents[agent].AgentCount = n_agents
    agents[agent].T_sim = t_sim
    agents[agent].track_vars = ['pLvl','t_age','Adjust',
                                'mNrm','nNrm','mNrmTilde','nNrmTilde','aNrm',
                                'cNrm', 'Share', 'DNrm']
    agents[agent].initialize_sim()
    agents[agent].simulate()
    profile = age_profiles(agents[agent])
    profile['Model'] = agent
    profiles.append(profile)

# %% Hark plot setup
from HARK.utilities import (
    determine_platform,
    test_latex_installation,
    setup_latex_env_notebook,
)

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

# %% Plot life-cycle means

simdata = pd.concat(profiles)

# Latex names
simdata = simdata.rename(columns = {'pLvl': 'Perm. Income $P$',
                 'Mtilde': 'Risk-free Assets $\\tilde{M}$',
                 'Ntilde': 'Risky Assets $\\tilde{N}$',
                 'C': 'Consumption  $C$',
                 'StockShare': 'Risky Share of Savings',
                 'Share': 'Deduct. Share $\\Contr$'})

lc_means = pd.melt(simdata,
                   id_vars = ['t_age', 'Model'],
                   value_vars = ['Perm. Income $P$',
                                 'Risk-free Assets $\\tilde{M}$',
                                 'Risky Assets $\\tilde{N}$',
                                 'Consumption  $C$',
                                 'Risky Share of Savings','Deduct. Share $\\Contr$'])

lc_means['Age'] = lc_means['t_age'] + 24

# Drop the last year, as people's behavior is substantially different.
lc_means = lc_means[lc_means['Age']<max(lc_means['Age'])]

# General aesthetics
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

g = sns.FacetGrid(
    lc_means,
    col="variable",
    col_wrap = 3,
    hue="Model",
    height=3,
    aspect=(7 / 3) * 1 / 3,
    sharey=False
)
g.map(sns.lineplot, "Age", "value", alpha=0.7)
g.add_legend(bbox_to_anchor=[0.5, 0.0], ncol=4, title="")
g.set_axis_labels("Age", "")
g.set_titles(col_template = '{col_name}')
