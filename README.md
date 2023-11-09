# Mapping Plant Nitrogen Using Hyperspectral and Machine Learning (Code Example)

**Welcome!**

This code sample demonstrates:
- Working with part of the remote sensing data that I specialize in (hyperspectral).
- Ensuring the reproducibility of the work.
- Applying a machine learning technique: Partial Least Square Regression (PLSR).

## Goal

The main goal of this project is to create a nitrogen map of shrublands in the western US using hyperspectral data. To achieve this, we use Partial Least Square Regression (PLSR), a machine learning technique. In this model, the spectral bands serve as predictors (features), and the measured percent nitrogen content is the target variable. [Here](https://github.com/hamiddashti/earthlab/blob/main/references/Short%20answer%20on%20why%20plsr.pdf) is a brief rationale for choosing PLSR for this project. The nitrogen data were collected from multiple shrubs within 10x10 meter plots in Idaho and California. The hyperspectral data is acquired from airborne AVIRIS-NG sensors. The original data is publicly available:

- [Field Data Link](https://daac.ornl.gov/VEGETATION/guides/Idaho_field_shrub_data.html)
- [Hyperspectral Images Link](https://daac.ornl.gov/VEGETATION/guides/AVIRIS-NG_Data_Idaho.html)

The code aims to showcase how machine learning can be utilized in remote sensing. Therefore, the field data used here is pre-processed and summarized in a CSV file, and the image is a small subset of the original data. Both files are available on Google Drive:

- [Field Data for This Project](https://drive.google.com/file/d/1UOEeyzHW-h0el2Qzk1o7BiSsqT8f8ax2/view?usp=sharing)
- [Image for This Project](https://drive.google.com/file/d/1XZMnMvglfqABTA3oVaJGUM3X-qV1uOqa/view?usp=sharing)

The provided scripts perform these steps:
1. Download the data.
2. Clean the training data.
3. Perform initial data exploration.
4. Train the PLSR model using cleaned data and two different feature selection methods.
5. Apply the better-performing model on the hyperspectral image to create the nitrogen map.

**Below is the project organization and instructions to reproduce the results.**

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README file overview the project.
    ├── data
    │   ├── processed      <- The final data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Documentation provided by Sphinx
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Main juypter notebook (00-hd-estimate-nitrogen.ipynb)
    │
    │
    ├── references         <- Includes some reference manuscripts and my response
    │
    ├── reports
    │   └── figures        <- Generated graphics and figures
    │
    ├── environment.yml   <- The conda environment generated with `conda env freeze > requirements.txt`
    │
    ├── src                <- Source code used in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Script to download data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models
    │   │   └──earhlab_lib.py <- Scripts to train model and then use it to
    │   │                           make predictions
    └───

## Steps to reproduce the results

1. **Clone this repository**

   git clone <https://github.com/hamiddashti/earthlab.git>

2. **Create a conda environment**

   conda env create -f environment.yml

3. **Lunch the Jupyter Notebook**

   jupyter notebook

4. Open the following notebook from notebook directory:

   cd notebooks

   [00-hd-estimate-nitrogen.ipynb](notebooks/00-hd-estimate-nitrogen.ipynb)

5. Run cells

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
