# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:47:33 2021

@author: Mateo
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def pol_funcs_dframe(agents, t, mNrmGrid, nNrm_vals, Share_vals):

    # Create tiled arrays for Reb and Sha stages
    m_tiled, n_tiled = np.meshgrid(mNrmGrid, nNrm_vals)
    
    dframes = []
    
    # Rebalancing
    for a in agents:
        dfrac = agents[a].solution[t].stage_sols['Reb'].dfracFunc_Adj(m_tiled, n_tiled)
        data = pd.DataFrame({'m': m_tiled.flatten(),
                             'n': n_tiled.flatten(),
                             'value': dfrac.flatten()})
        data['model'] = a
        data['control'] = 'dfrac'
        
        dframes.append(data)
    
    # Contrib share
    for a in agents:
        s = agents[a].solution[t].stage_sols['Sha'].ShareFunc_Adj(m_tiled, n_tiled)
        data = pd.DataFrame({'m': m_tiled.flatten(),
                             'n': n_tiled.flatten(),
                             'value': s.flatten()})
        data['model'] = a
        data['control'] = 'Share'
        
        dframes.append(data)
    
    # Consumption
    m_tiled, n_tiled, Share_tiled = np.meshgrid(mNrmGrid, nNrm_vals, Share_vals)
    for a in agents:
        c = agents[a].solution[t].stage_sols['Cns'].cFunc(m_tiled, n_tiled, Share_tiled)
        data = pd.DataFrame({'m': m_tiled.flatten(),
                             'n': n_tiled.flatten(),
                             'Share': Share_tiled.flatten(),
                             'value': c.flatten()})
        data['model'] = a
        data['control'] = 'c'
        
        dframes.append(data)
    
    
    return pd.concat(dframes)

# %% Define a plotting function

def age_profiles(agent):

    # Flatten variables
    Data = {k: v.flatten(order = 'F') for k, v in agent.history.items()}

    # Make dataframe
    Data = pd.DataFrame(Data)

    Data['savingNrm'] = Data.aNrm + Data.nNrmTilde
    Data['StockShare'] = Data.nNrmTilde / Data.savingNrm

    # Create non-normalized versions of variables
    Data['M'] = Data.mNrm * Data.pLvl
    Data['N'] = Data.nNrm * Data.pLvl
    Data['C'] = Data.cNrm * Data.pLvl
    Data['Mtilde'] = Data.mNrmTilde * Data.pLvl
    Data['Ntilde'] = Data.nNrmTilde * Data.pLvl
    
    # Find the mean of each variable at every age
    AgeMeans = Data.groupby(['t_age']).mean().reset_index()

    return AgeMeans