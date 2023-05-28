# README #

## About this repository ##

This is the repository used for FRI Data Science Project Competition. The authors, Leon Hvastja and Amadej Pavšič, are working on feature selection in sparse datasets with Outbrain. 

More specifically, our goal is to test different feature ranking algorithms, in order to speed up subsequential processes. Ground truth for our testing is Outbrain's in-house developed autoML, which is accurate, but relatively slow.  

## Reproducibility

All code in `/src` should be reproducible. You need venv with Python 3.10.8 and install all packages from `/src/requirements.txt`.

NOTE! The data is not included in this repository, as it is not publicly available and is not ours to distribute. Furthermore, the data is too large to be stored in this repository.
So please acquire the data from Outbrain and place it in `/src/data` folder (either `full_data.csv.zip` or `full_data.csv`).

## Folder structure ##

There are several subfolders in the repository:

* the source folder (`/src`),
* the journal folder (`/journal`),
* the interim report folder (`/interim_report`),
* the final report folder (`/final_report`),
* and the presentation folder (`/presentation`).

### The source folder ###

Contains code & stuff.

* `EDA.ipynb`
  * Code and visualizations for exploratory data analysis

* `helper_functions.py`
  * various helper functions for feature ranking algorithms and evaluation pipeline

* `rank_eval_pipeline.py`
  * pipeline for evaluating feature ranking algorithms

* `ranking_algos_comparison_chosen.ipynb`
  * visualizations of feature ranking algorithms, comparison of chosen algorithms (different algorithms, subsets of data)

* `optimization_graphing.ipynb`
  * visualizations of subsampling optimization, ensemble technique

* `performance_graphing.ipynb`
  * visualizations of time vs performance of feature ranking algorithms

* data subfolder (`/data`)
    * placeholder for full_data.csv
    * contains ground truth data

* documentation subfolder (`/docs`)
  * feature descriptions
  * documentation of algorithm testing

* results subfolder (`/img`)
  * various visualizations of results (dendrograms, plots, etc.)

* results subfolder (`/results`)
  * various results of algorithm testing

* fuji subfolder (`/fuji`)
  * contains [Fuzzy Jaccard Index package](https://github.com/Petkomat/fuji-score), that was used as an experiment 
  to evaluate feature ranking algorithms. The package also contains some helper functions for feature ranking.


### The journal folder ###

Contains journal files of authors with description and quantification of work done.

### The interim report folder ###

Contains interim report as a LaTeX project and compiled PDF.

### The final report folder ###

Contains final report as a LaTeX project and compiled PDF.

### The presentation folder ###

Contains presentation in PDF and pptx format.