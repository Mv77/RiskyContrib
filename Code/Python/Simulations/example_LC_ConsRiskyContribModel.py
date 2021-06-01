
"""
Example implementations of HARK.ConsumptionSaving.ConsPortfolioModel
"""
from HARK.ConsumptionSaving.ConsRiskyContribModel import (
    RiskyContribConsumerType,
    init_risky_contrib_lifecycle,
)
from time import time
from copy import copy

import numpy as np
import seaborn as sns
import pandas as pd
from tools import pol_funcs_dframe, age_profiles
import os

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
# %% Create and solve agents with all the parametrizations

agents = {
    "Base": RiskyContribConsumerType(**par_LC_base),
    "Tax": RiskyContribConsumerType(**par_LC_tax),
    "Calvo": RiskyContribConsumerType(**par_LC_calvo),
    "Retirement": RiskyContribConsumerType(**par_LC_retirement),
}

#agents = {
#    "Retirement": RiskyContribConsumerType(**par_LC_retirement)
#}

for agent in agents:

    print("Now solving " + agent)
    t0 = time()
    agents[agent].solve(verbose=True)
    t1 = time()
    print("Solving " + agent + " took " + str(t1 - t0) + " seconds.")
    
# %% Solve and simulate
n_agents = 50
t_sim    = 5000
profiles = []
for agent in agents:
    agents[agent].AgentCount = n_agents
    agents[agent].T_sim = t_sim
    agents[agent].track_vars = ['pLvl','t_age','Adjust',
                                'mNrm','nNrm','mNrmTilde','nNrmTilde','aNrm',
                                'cNrm', 'Share', 'dfrac']
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


def make(fig, name, target_dir="../../../Figures"):
    fig.savefig(os.path.join(target_dir, "{}.pdf".format(name)))

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
make(g, "LC_age_profiles")