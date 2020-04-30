# rllib

To install on clusters do:
conda create -n rllib python=3.6
conda activate rllib 

pip install -e .[test,logging]
sudo apt-get install -y --no-install-recommends --quiet build-essential libopenblas-dev python-opengl xvfb xauth
