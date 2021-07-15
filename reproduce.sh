# requirements setup and mamba for faster install for env
source /opt/conda/etc/profile.d/conda.sh
mamba env create -qq -f environment.yml
conda activate RiskyContrib

# Then pull a version of HARK from GitHub. It has not been
# released yet
#pip install git+git://github.com/econ-ark/HARK

# execute the script to create figures
ipython do_ALL.py
