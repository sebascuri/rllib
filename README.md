# rllib

[![CircleCI](https://circleci.com/github/sebascuri/rllib/master.svg?style=svg&circle-token=ff0b332138b08cf89b759461b55827a3eec18390)](https://app.circleci.com/pipelines/github/sebascuri/rllib)

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