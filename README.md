# Noise Covariance Estimation for Multi-Task Linear Models

This repository provides the source code for the paper 
[Noise covariance estimation in multi-task high-dimensional linear models](https://projecteuclid.org/journals/bernoulli/volume-30/issue-3/Noise-covariance-estimation-in-multi-task-high-dimensional-linear-models/10.3150/23-BEJ1644.short)

## Description

Description of files and folder:

* **lib_using_einsum.py** : Computes the interaction matrix, will be used in **main.py**.
* **main.py** : Calculates estimators for noise covariance matrix, saves results into folder **EstimationResults**.
* **run_boxplot.py**: Generates and saves *boxplots* into folder **EstimationResults**.
* **run_heatmap.py**: Generates and saves *heatmaps* into folder **EstimationResults**.

## Connect simulation setup to our code

In the manuscript, for different sample size $n$ (1000, 1500, 2000), we conducted simulations for estimating two types of noise covariance matrices (full-rank and low-rank). 

In the Python script **main.py**, we can specify these simulation setups by variables $n$ taking value from {1000, 1500, 2000}, and `S_type` taking values in {`full_rank`, `low_rank`}. 

## Executing steps

Executing the following steps will generate all the figures used in the manuscript. 

1. Run **main.py** with all the six combinations of $n$ and `S_type`: this step will compute and save all the estimators into the folder **Estimation_Results**.
3. Run **run_boxplot.py**: this step will generate boxplots (Figure 1 in the full paper) and save into the folder **Estimation_Results**.
4. Run **run_heatmap.py**: this step will generate heatmaps (Figure 2 in the full paper and Figures 3-7 in the supplementary material) and save into the folder **Estimation_Results**.

## Hardware
All the simulations were run on a cluster of 50 CPU-cores (each is an Intel Xeon E5-2680 v4 @2.40GHz) equipped with a total of 150 GB of RAM. 

## Execution time
* We used Python 3.9.6 to run all the simulations. 
* Most of the execution time is spent running the script **main.py**, while other steps cost only a few seconds.
* For each choice of the `S_type`, running **main.py** takes approximately 30, 60, 100 minutes for $n=1000, 1500, 2000$, respectively. 
* In total, running all the simulations with different sample size $n$ and `S_type` takes about 6 hours. 

## Citation
```
@article{tan2024noise,
  title={Noise covariance estimation in multi-task high-dimensional linear models},
  author={Tan, Kai and Romon, Gabriel and Bellec, Pierre C},
  journal={Bernoulli},
  volume={30},
  number={3},
  pages={1695--1722},
  year={2024},
  publisher={Bernoulli Society for Mathematical Statistics and Probability}
}
```
