Nitrogen Map
==============================

Welcome! And thanks for the oppurtunity.

Here is the code sample that demostrate:

* Part of remote sensing data that I work with (hyperspectral).
* The reproducibility of the work.
* A machine learning technique (partial least square regression) application.

**Below you can find the project organization, description and how to reproduce the results.**

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README file overview the project.
    ├── data
    │   ├── processed      <- The final data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-hd-initial-data-exploration`.
    │
    ├── references         <- Includes some reference manuscripts
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures
    │
    ├── environment.yml   <- The conda environment 
    │                        generated with `conda env freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Script to download data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    └─── 

Goal
------------

The main goal of this project is to create the nitrogen map of shrublands in western US using spectral imaging. To do so, we use partial least square regression (PLSR) which is a machine learning technique. In this model the spectral bands are the predictors (features) and the measured percent nitrogen content is the target variable. The nitrogen data were measured from multiple shrubs in 10*10 meters plot collected from Idaho and California. The hyperspectral data comes from airborne AVIRIS-NG sensor. The original data is publically available:

* [Link to the filed data](https://daac.ornl.gov/VEGETATION/guides/Idaho_field_shrub_data.html)
* [Link to the hyperspectral images](https://daac.ornl.gov/VEGETATION/guides/AVIRIS-NG_Data_Idaho.html)

The objective of this code is a presentation of one of the ways that machine learning can be used in remote sensing. Thus, the field data used in this project is already processed and summarized in a csv file. The image is also an small subsample of the original data. Both files are uploaded on Google Drive:

* [Link to the field data used in this study](https://drive.google.com/file/d/1UOEeyzHW-h0el2Qzk1o7BiSsqT8f8ax2/view?usp=sharing)
* [Link to the image](https://drive.google.com/file/d/1XZMnMvglfqABTA3oVaJGUM3X-qV1uOqa/view?usp=sharing)

The scripts provided in this project perform the following steps:

1. Download the data.
2. Clean the training data.
3. Some initial data exploration.
4. Train the PLSR model using cleaned data and two different feature selection methods.
5. Apply the model with better performance on the hyperspectral image to create the percent nitrogen map.

Steps to reproduce the results
----------

1. **Clone this repository**

    git clone <https://github.com/hamiddashti/earthlab.git>
2. **Create a conda environment**
    
    conda env create -f environment.yml

3. **Lunch the Jupyter Notebook**

    jupyter notebook
4. Open the following notebook:

    [00-hd-estimate-nitrogen.ipynb](../notebooks/00-hd-estimate-nitrogen.ipynb)
    
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
