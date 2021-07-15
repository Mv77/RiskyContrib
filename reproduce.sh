# requirements setup and mamba for faster install for env
source /opt/conda/etc/profile.d/conda.sh
mamba env create -qq -f environment.yml
conda activate RiskyContrib

# execute the script to create figures
ipython do_ALL.py
