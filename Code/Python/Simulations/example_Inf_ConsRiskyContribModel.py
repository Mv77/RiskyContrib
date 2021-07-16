# %%
"""
Example implementations of HARK.ConsumptionSaving.ConsPortfolioModel
"""
from HARK.ConsumptionSaving.ConsRiskyContribModel import (
    RiskyContribConsumerType,
    rebalance_assets,
    init_risky_contrib,
)
from time import time
from copy import copy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    # Running as a script
    from tools import pol_funcs_dframe
else:
    # Running from do_ALL
    from Simulations.tools import pol_funcs_dframe

# %% Path for figures

if __name__ == '__main__':
    # Running as a script
    my_file_path = os.path.abspath("../../../")
else:
    # Running from do_ALL
    my_file_path = os.path.dirname(os.path.abspath("do_ALL.py"))

FigPath = os.path.join(my_file_path,"Figures/")

# %% Base parametrization

# Make the problem infinite horizon
par_infinite_base = init_risky_contrib.copy()
par_infinite_base["cycles"] = 0

# and frictionless to start
par_infinite_base["AdjustPrb"] = 1.0
par_infinite_base["tau"] = 0.0

# Make contribution shares a continuous choice
par_infinite_base["DiscreteShareBool"] = False
par_infinite_base["vFuncBool"] = False

# Make grids coarser to improve runtime.
par_infinite_base.update(
    {"aXtraCount": 30, "mNrmCount": 30, "nNrmCount": 30,}
)

# %% A version with the tax
par_infinite_tax = copy(par_infinite_base)
par_infinite_tax["tau"] = 0.1

# %% A version with the Calvo friction
par_infinite_calvo = copy(par_infinite_base)
par_infinite_calvo["AdjustPrb"] = 0.25

# %% Create and solve agents with all the parametrizations

agents = {
    "Base": RiskyContribConsumerType(**par_infinite_base),
    "Tax": RiskyContribConsumerType(**par_infinite_tax),
    "Calvo": RiskyContribConsumerType(**par_infinite_calvo),
}

for agent in agents:

    print("Now solving " + agent)
    t0 = time()
    agents[agent].solve(verbose=True)
    t1 = time()
    print("Solving " + agent + " took " + str(t1 - t0) + " seconds.")

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


def make(fig, name, target_dir=FigPath):
    fig.savefig(os.path.join(target_dir, "{}.pdf".format(name)))


# %% Plots


t = 0
mNrmGrid = np.linspace(0, 40, 100)
nNrm_vals = np.array([0.0, 20.0, 40])
Share_vals = np.array([0.0, 0.5])

polfuncs = pol_funcs_dframe(agents, t, mNrmGrid, nNrm_vals, Share_vals)

# General aesthetics
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


# Rebalancing fraction
polfuncs["$n$"] = polfuncs["n"]
g = sns.FacetGrid(
    polfuncs[polfuncs.control == "dfrac"],
    col="$n$",
    hue="model",
    height=3,
    aspect=(7 / 3) * 1 / 3,
)
g.map(sns.lineplot, "m", "value", alpha=0.7)
g.add_legend(bbox_to_anchor=[0.5, 0.0], ncol=3, title="")
g.set_axis_labels("$m$", "Rebalancing fraction: $\dFrac$")
make(g, "inf_dFunc")
plt.pause(1)

# After rebalancing, m and n turn to their "tilde" versions. Create ntilde
# just for seaborn's grid labels.
polfuncs["$\\tilde{n}$"] = polfuncs["n"]
polfuncs["$\\Contr$"] = polfuncs["Share"]

# Share fraction
g = sns.FacetGrid(
    polfuncs[polfuncs.control == "Share"],
    col="$\\tilde{n}$",
    hue="model",
    height=3,
    aspect=(7 / 3) * 1 / 3,
)
g.map(sns.lineplot, "m", "value", alpha=0.7)
g.add_legend(bbox_to_anchor=[0.5, 0.0], ncol=3, title="")
g.set_axis_labels("$\\tilde{m}$", r"Deduction Share: $\Contr$")

make(g, "inf_ShareFunc")
plt.pause(1)

# Consumption fraction
g = sns.FacetGrid(
    polfuncs[polfuncs.control == "c"],
    col="$\\tilde{n}$",
    row="$\\Contr$",
    hue="model",
    height=3,
    aspect=(7 / 3) * 1 / 3,
)
g.map(sns.lineplot, "m", "value", alpha=0.7)
g.add_legend(bbox_to_anchor=[0.5, 0.0], ncol=3, title="")
g.set_axis_labels("$\\tilde{m}$", "Consumption: $c$")

make(g, "inf_cFunc")
plt.pause(1)

# %% Rebalancing viz

# Create a grid of (m,n)
max_assets = 50
npoints = 6
m_tiled, n_tiled = np.meshgrid(np.linspace(0, max_assets, npoints),
                               np.linspace(0, max_assets, npoints))

import matplotlib.pyplot as plt


for i in range(len(agents)):
    
    name = list(agents.keys())[i]
    
    d = agents[name].solution[0].stage_sols["Reb"].dfracFunc_Adj(m_tiled, n_tiled)
    
    mTil_tiled, nTil_tiled = rebalance_assets(d, m_tiled, n_tiled,
                                              agents[name].tau)
    
    g = plt.figure()
    plt.quiver(m_tiled, n_tiled,
                  mTil_tiled - m_tiled, nTil_tiled - n_tiled,
                  units='xy', angles = "xy", scale = 1, linewidths = 2)
    
    plt.plot(m_tiled, n_tiled, '.k')
    plt.plot(mTil_tiled, nTil_tiled, 'xr')
    
    plt.xlim(-1, max_assets + 1)
    plt.ylim(-1, max_assets + 1)
    plt.title('Rebalancing in the ' + name + ' model')
    plt.xlabel('Risk-free assets $m$ and $\\tilde{m}$')
    plt.ylabel('Risky assets $n$ and $\\tilde{n}$')

    make(g, "inf_rebalance_"+name)
    plt.pause(1)