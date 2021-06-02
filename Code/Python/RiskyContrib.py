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

from HARK.ConsumptionSaving.ConsRiskyContribModel import (
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
#
# I now discuss the main components of the model informally, and leave its full
# recursive mathematical representation for Section \ref{sec:recursive}.
#
# ### Time, mortality, and utility
#
# Time advances in discrete steps that I will index with $t$. The model can
# be used in both infinite and finite-horizon versions.
#
# Agents face an exogenous risk of death $\delta_t$ each period, which becomes certain at the 
# maximum age of the finite-horizon version. There are no intentional bequests; agents
# will consume all of their resources if they reach the last period, but they can leave
# accidental bequests upon premature death.
#
# In each period, agents derive utility from consumption only. Their utility function
# follows a constant relative risk aversion specification. Formally, for a level of 
# consumption $C$, the agent derives instant utility
# \begin{equation}
# 	u(C) = \frac{C^{1-\rho}}{1- \rho}.
# \end{equation}
#
# #### Income process
#
# Agents supply labor inelastically. Their labor earnings $Y_{i,t}$ are the product of a
# permanent component $P_{i,t}$ and a transitory stochastic component $\theta_{i,t}$ as
# in \cite{Carroll1997qje},  where $i$ indexes different agents. Formally,
# \begin{equation*}
# \begin{split}
# \ln Y_{i,t} &= \ln P_{i,t} + \ln \theta_{i,t} \\
# \ln P_{i,t} &= \ln P_{i,t-1} + \ln \Gamma_{i,t} + \ln \psi_{i,t}
# \end{split}
# \end{equation*}
# where $\Gamma_{i,t}$ is a deterministic growth factor that can capture
# life-cycle patterns in earnings, and
# $\ln \psi_{i,t}\sim \mathcal{N}(-\sigma^2_{\psi,t}/2, \sigma_{\psi,t})$
# is a multiplicative shock to permanent income\footnote{The mean of the shock is set so that $E[\psi_{i,t}] = 1$.}.
#
# The transitory component $\theta_{i,t}$ is a mixture that models unemployment and
# other temporal fluctuations in income as
# \begin{equation*}
# \ln\theta_{i,t} = \begin{cases}
# \ln \mathcal{U}, & \text{With probability } \mho\\
# \ln \tilde{\theta}_{i,t}\sim\mathcal{N}(-\sigma^2_{\theta,t}/2, \sigma_{\theta,t}), & \text{With probability } 1-\mho,
# \end{cases}
# \end{equation*}
# with $\mho$ representing the probability of unemployment and $\mathcal{U}$ the replacement
# factor of unemployment benefits.
#
# This specification of the income process is parsimonious and flexible enough to accommodate
# life-cycle patterns in income growth and volatility, transitory unemployment and exogenous
# retirement. Introduced by \cite{Carroll1997qje}, this income specification is common in studies
# of life-cycle wealth accumulation and portfolio choice; see e.g.,
# \cite{Cagetti2003jbes,Cocco2005rfs,Fagereng2017jof}. The specification has
# also been used in studies of income volatility, which have yielded calibrations of its stochastic
# shocks' distributions \citep[see e.g.,][]{Carroll1992bpea,Carroll1997jme,Sabelhaus2010jme}
#
# #### Financial assets and frictions
#
# Agents smooth their consumption by saving and have two assets
# available for this purpose. The first is a risk-free liquid account with 
# constant per-period return factor $R$. The second has a stochastic return
# factor $\tilde{R}$ that is log-normally distributed and independent across
# time. Various interpretations such as stocks, a retirement fund, or entrepreneurial
# capital could be given to the risky asset. Importantly, consumption must be paid for
# using funds from the risk-free account. The levels of risk-free and risky assets
# owned by the agent will both be state variables, denoted with $M_{i,t}$ and $N_{i,t}$
# respectively.
#
# Portfolio rebalancing takes place by moving funds between the risk-free
# and risky accounts. These flows are one of the agents' control variables
# and are denoted as $D_{i,t}$, with $D_{i,t}>0$ representing a movement of
# funds from the risk-free to the risky account. Withdrawals from the risky
# account are subject to a constant-rate tax $\tau$ which can represent, for
# instance, capital-gains realization taxes or early retirement-fund withdrawal
# penalties. In sum, denoting post-rebalancing asset levels with $\tilde{\cdot}$,
# \begin{equation*}
# \begin{split}
# \tilde{M}_{i,t} &= M_{i,t} - D_{i,t}(1 - 1_{[D_{i,t}\leq0]}\tau)\\
# \tilde{N}_{i,t} &= N_{i,t} + D_{i,t}.
# \end{split}
# \end{equation*}
#
# At any given period, an agent might not be able to rebalance his portfolio.
# This ability is governed by an exogenous stochastic shock that is realized
# at the start of the period
# \begin{equation*}
# \Adj_t \sim \text{Bernoulli}(p_t),
# \end{equation*}
# with $\Adj_t=1$ meaning that the agent can rebalance and $\NAdj_t=1$ ($\Adj_t = 0$)
# forcing him to set $D_{i,t} = 0$. This friction is a parsimonious way to capture
# the fact that portfolio rebalancing is costly and households do it sporadically.
# Recent studies have advocated for \citep{Giglio2021aer} and used
# \citep{Luetticke2021aej_macro} this kind of rebalancing friction.
#
# To partially evade the possibility of being unable to rebalance their accounts, agents
# can use an income deduction scheme. By default, labor income ($Y_{i,t}$) is deposited to
# the risk-free liquid account at the start of every period. However, agents can pre-commit
# to have a fraction  $\Contr_t\in[0,1]$ of their income diverted to their risky account instead.
# This fraction can be tweaked by the agent whenever $\Adj_t = 1$; otherwise it stays at its
# previous value, $\Contr_{t+1} = \Contr_t$.
#
# #### Timing
#
# \input{\FigDir/Timing_diagram}
#
# Figure \ref{fig:Timing_diagram} summarizes the timing of stochastic shocks and
# optimizing decisions that occur within a period of the life cycle model.
#
# ### Recursive representation of the model}
#
# Individual subscripts $i$ are dropped for simplicity. The value function for
# an agent who is not allowed to rebalance his portfolio at time $t$ is
# \begin{equation*}
#     \input{\EqDir/bellman_NAdj}
# \end{equation*}
# and that of agent who is allowed to rebalance is
# \begin{equation*}
#     \input{\EqDir/bellman_Adj}
# \end{equation*}

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
    {"aXtraCount": 25, "mNrmCount": 25, "nNrmCount": 25,
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
