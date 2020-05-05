# rllib

[![CircleCI](https://circleci.com/gh/sebascuri/rllib/master.svg?style=shield&circle-token=25c056fd6b7e322c55dd48fd0c6052b1f8800919)](https://app.circleci.com/pipelines/github/sebascuri/rllib)

To install on clusters do:
conda create -n rllib python=3.6
conda activate rllib 

pip install -e .[test,logging]
sudo apt-get install -y --no-install-recommends --quiet build-essential libopenblas-dev python-opengl xvfb xauth



To run locally circleci run:
```bash
$ circleci config process .circleci/config.yml > process.yml
$ circleci local execute -c process.yml --job test
```