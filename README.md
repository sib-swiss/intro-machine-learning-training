June 2022 course: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6777795.svg)](https://doi.org/10.5281/zenodo.6777795)

# Introduction to Machine Learning (with python)

This repository regroups works on the "introduction to machine learning" course of the SIB.
Its goal is to develop a somewhat language agnostic course, with a an R and a python implementation, however at the moment only the python implementation in done.

## pre-requisites

The course is targeted to life scientists who are already familiar with the Python programming language and who have basic knowledge on statistics.

In order to follow the course you need to have installed [python](https://www.python.org/) and [jupyter notebooks](https://www.jupyter.org/) (a web based notebook system for creating and sharing computational documents). 

If you need help with the installation, you can refer to these [tips and instructions](https://github.com/sib-swiss/first-steps-with-python-training/blob/master/setting_up_your_environment.md)(NB: this links to another github repo).

In addition, please ensure you have the following libraries installed:
 * [seaborn](https://seaborn.pydata.org/installing.html)
 * [scikit-learn](https://scikit-learn.org/stable/install.html)

> NB: we will use several other libraries, such as numpy, scipy, matplotlib, or pandas, but they are pre-requisites of the 2 above, so they should be installed automatically if you install using pip or conda.


## course organization 

The course is organized in several, numbered, jupyter notebooks, each corresponding to a chapter which interleaves theory, code demo, and exercises.

The course does not require any particular expertise with jupyter notebooks to be followed, but if it is the first time you encounter them we recommend this [gentle introduction](https://realpython.com/jupyter-notebook-introduction/).

 * [Chapter0 : python warmup](python_notebooks/Chapter_0_python_warmup.ipynb): provides a gentle warm-up to the basic usage of the libraries which are a pre-requisite for this course. You can use this notebook to assess your level before the course, or just as a way to get (re-)acquainted with these libraries.
 * [Chapter1 : Exploratory analysis and unsupervised learning](python_notebooks/Chapter_1_Exploratory_analysis_and_unsupervised_learning.ipynb)
 * [Chapter2 : Machine Learning routine - distance-based model for classification](python_notebooks/Chapter_2_Machine_Learning_routine__distance_based_model_for_classification.ipynb)
 * [Chapter3 : Machine Learning based on decision trees for classification](python_notebooks/Chapter_3_Machine_Learning_based_on_decision_trees_for_classification.ipynb)
 * [Chapter4 : Machine Learning for regression](python_notebooks/Chapter_4_Machine_Learning_for_regression.ipynb)
 * [Chapter Extra : Teasing Neural Network](python_notebooks/Chapter_Extra_Teasing_Neural_Network.ipynb)

Solutions to each practical can be found in the [`solutions/`](python_notebooks/solutions/) folder and should be loadable directly in the jupyter notebook themselves.

Note also the `utils.py` and `utils2.py` files which contain many utilitary function we use to visually showcase the effect of various ML algorithms' hyper-parameters.



## directory structure


* data : contains the datasets
* exam : contains the data and instruction of a facultative exam
* images : images generated or used in the notebooks
* python_notebooks
* references : reference to the article of some of the datasets
* slides : pptx / pdf of introductory slides 


## course material citation

Please cite as : 

Markus Mueller, Wandrille Duchemin, & Patricia Palagi. (2022, June 29). sib-swiss/intro-machine-learning-training: June 2022 course edition. Zenodo. [https://doi.org/10.5281/zenodo.6777795](https://doi.org/10.5281/zenodo.6777795)


## TODO

todo list :
 * "translate" python notebooks to R
 	* chapter 1
 	* chapter 2
 	* chapter 3
 	* chapter 4

