# Environment setup for the introduction to Machine Learning Course

> :exclamation: if you encounter any error with the instructions given on this page, please create a [github issue](https://github.com/sib-swiss/intro-machine-learning-training/issues/new) to explain your problem and we will try to get back to you ASAP.


We detail in this page how to set up your environment with the different external modules you will need in order to be able to follow the course.

We recommend you create a new conda environment specifically for the course (if you are unfamiliar with conda environment see [this documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)). 

Nevertheless, we detail here several methods a trust you will choose the one most appropriate to your situations.

**important**: the course materials were developped and tested with **python >=3.12 and scikit-learn >=1.8**. Any anterior version will give errors and warnings aplenty!

> NB: for future reference, some previous versions of this course were run with python 3.8 and scikit-learn 1.0.1 (prior to 2022), or python 3.10.5 and scikit-learn 1.1.1 (2022 to 2024)


## method 1 : conda and pip commands to install 

If you are on Windows and/or are allergic to command line, you can use the [anaconda navigator](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/#importing-an-environment) (if you don't know how to start the navigator, [here's how](https://docs.anaconda.com/anaconda/navigator/getting-started/#starting-navigator)).


Otherwise you can use the commande line to create and activate a new enviroment
```
conda create -y -n introML python=3.12
conda activate introML
```

These commands install all necessary modules and their dependencies:
```
conda install -c conda-forge -y scikit-learn seaborn xlrd openpyxl umap-learn plotly ipywidgets ipykernel nbformat
pip install string-kernels
conda install -c pytorch -y pytorch torchvision torchaudio cpuonly 
```

## method 2 : create a uv project

If you want to manage the environment for this course using [uv](https://docs.astral.sh/uv) (a fast python project and package manager), you can use the following commands:

```
uv init . # presuming your terminal is already in the folder where you want the project
uv add scikit-learn seaborn xlrd openpyxl umap-learn plotly string-kernels ipywidgets ipykernel nbformat jupyterlab
uv add torch
```

You can then launch a jupyterlab server with:
```
uv run --with jupyter jupyter lab
```

## method 3 : install the following however you want

Python : at least 3.12

 * scikit-learn (version ar least 1.8)
 * jupyter-notebook
 * seaborn
 * xlrd
 * openpyxl
 * umap-learn
 * plotly
 * ipywidgets
 * nbformat
 * string-kernels
 * pytorch
