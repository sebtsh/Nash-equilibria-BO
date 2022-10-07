# No-regret Sample-efficient Bayesian Optimization for Finding Nash Equilibria with Unknown Utilities

This code repository accompanies the paper "No-regret Sample-efficient Bayesian Optimization for Finding Nash Equilibria 
with Unknown Utilities".

## Requirements
1. Linux machine (experiments were run on Ubuntu 20.04.4 LTS)
2. Python >= 3.7
3. NumPy >= 1.21.5 (scipydirect requires NumPy to be installed first)
4. A Fortran compiler for scipydirect (if using Anaconda, this can be easily installed with ` conda install -c conda-forge gfortran `)

## Setup
In the main directory, run the following command to install the required libraries.
```shell
pip install -r requirements.txt
```

## Experiments
To run individual experiments for specific seeds, the following commands may be used for pure
and mixed NE respectively. `GAME` can be one of {`rand`, `gan`, `bcad`}. `ACQ` can be one of
{`ucb_pne`, `ucb_pne_noexplore`, `prob_eq`, `BN`} for pure NE, and one of 
{`ucb_mne`, `ucb_mne_noexplore`, `max_ent_mne`, `random_mne`} for mixed NE. `SEED` can
be any integer.
```shell
python pnebo.py with {GAME} acq_name={ACQ} seed={SEED}
```
```shell
python mnebo.py with {GAME} acq_name={ACQ} seed={SEED}
```
Alternatively, to run experiments in the background with seeds in the range `START`-`END` (inclusive),
the following command may be used. `TYPE` can be one of {`pne`, `mne`}.
```shell
./run.sh {TYPE} {GAME} {ACQ} {START} {END}
```
To run the timing experiments for pure NE, run the following command:
```shell
python timing.py with {GAME}
```


## Results
Once the experiments have completed, the following commands may be
used to plot the regrets. The seeds used must be put in `pne_results.py`
and `mne_results.py`, otherwise they will assume you used the seeds declared
in the paper.
```shell
python pne_results.py with {GAME}
```
```shell
python mne_results.py with {GAME}
```