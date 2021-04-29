# %%
'''
Example implementations of HARK.ConsumptionSaving.ConsPortfolioModel
'''
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyContribConsumerType, init_riskyContrib
from time import time

from tools import plotSlices3D, plotSlices4D

# %% Base parametrization

# Make the problem infinite horizon
par_infinite_base = init_riskyContrib.copy()
par_infinite_base['cycles']   = 0

# and frictionless to start
par_infinite_base['AdjustPrb'] = 1.0
par_infinite_base['tau'] = 0.0

# Make contribution shares a continuous choice
par_infinite_base['DiscreteShareBool'] = False
par_infinite_base['vFuncBool'] = False

# %%
# Solve base version

# Create agent and solve it.
inf_base_agent = RiskyContribConsumerType(**par_infinite_base)
print('Now solving base version')
t0 = time()
inf_base_agent.solve(verbose = True)
t1 = time()
print('Converged!')
print('Solving took ' + str(t1-t0) + ' seconds.')

# Plot policy functions
periods = [0]
n_slices = [0,2,6]
mMax = 20

DFuncAdj     = [inf_base_agent.solution[t].stageSols['Reb'].DFuncAdj for t in periods]
ShareFuncSha = [inf_base_agent.solution[t].stageSols['Sha'].ShareFuncAdj for t in periods]
cFuncFxd     = [inf_base_agent.solution[t].stageSols['Cns'].cFunc for t in periods]

# Rebalancing
plotSlices3D(DFuncAdj,0,mMax,y_slices = n_slices,y_name = 'n',
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m','d'])
# Share
plotSlices3D(ShareFuncSha,0,mMax,y_slices = n_slices,y_name = 'n',
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m','S'])

# Consumption
shares = [0., 0.9]
plotSlices4D(cFuncFxd,0,mMax,y_slices = n_slices,w_slices = shares,
             slice_names = ['n_til','s'],
             titles = ['t = ' + str(t) for t in periods],
             ax_labs = ['m_til','c'])