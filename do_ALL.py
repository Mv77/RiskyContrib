# Runtime ~1500 seconds
import sys
sys.path.append('./Code/Python/')

import time 

# %% Set up plot displays
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')

# %% PDF's results

# Start timer
start = time.process_time()

# 1. Solve infinite horizon versions of the model and display their policy
#    functions.
print('1. Solve infinite horizon versions')
import Simulations.example_Inf_ConsRiskyContribModel

# 2. Solve life-cycle versions of the model and use them to simulate
#    populations. Inspect the average live-cycle profiles of variables of
#    interest.
print('2. Solve and simulate life-cycle versions.')
import Simulations.example_LC_ConsRiskyContribModel

# Print time
print('Runtime was {} minutes'.format((time.process_time() - start))/60)