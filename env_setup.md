# Environment setup for the introduction to Machine Learning Course

> :exclamation: if you encounter any error with the instructions given on this page, please create a [github issue](https://github.com/sib-swiss/intro-machine-learning-training/issues/new) to explain your problem and we will try to get back to you ASAP.


We detail in this page how to set up your environment with the different external modules you will need in order to be able to follow the course.

We recommend you create a new conda environment specifically for the course (if you are unfamiliar with conda environment see [this documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)). 

Nevertheless, we detail here several methods a trust you will choose the one most appropriate to your situations.

**important**: the course materials were developped and tested with **python >=3.11 and scikit-learn >=1.5**. Any anterior version will give errors and warnings aplenty!

> NB: for future reference, some previous versions of this course were run with python 3.8 and scikit-learn 1.0.1 (prior to 2022), or python 3.10.5 and scikit-learn 1.1.1 (2022 to 2024)



## method 1 : new conda environment from `.yml`

Download the file <a href="https://downgit.github.io/#/home?url=https://github.com/sib-swiss/intro-machine-learning-training/blob/main/introML.yml" targte="_blank">introML2022.yml</a>.


If you are on Windows and/or are allergic to command line, you can use the [anaconda navigator](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/#importing-an-environment) (if you don't know how to start the navigator, [here's how](https://docs.anaconda.com/anaconda/navigator/getting-started/#starting-navigator)).


Otherwise, just open a terminal, navigate to where the file is, and use the following command:
```
conda env create -f introML.yml
```

Activate the new environment: `conda activate introML`

Verify that the new environment was installed correctly: `conda env list`

## method 2 : conda and pip commands to install 

These first 2 commands create and activate a new enviroment
```
conda create -y -n introML python=3.11
conda activate introML
```

These commands install all necessary modules and their dependencies:
```
conda install -y scikit-learn seaborn xlrd openpyxl umap-learn 
conda install -y plotly ipywidgets
pip install string-kernels
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
```


## method 3 : install the following however you want

Python : at least 3.11

 * scikit-learn (version ar least 1.8)
 * jupyter-notebook
 * seaborn
 * xlrd
 * openpyxl
 * umap-learn
 * plotly
 * ipywidgets
 * string-kernels
 * pytorch
