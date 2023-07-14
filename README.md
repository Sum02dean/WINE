# WINE
Wine classification project
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg?style=flat)](https://github.com/feross/standard)

<p align="center">
  <img src="wine_img.png" width="800">
</p>


## Setup
Dependencies for conda and pip are listed in `environment.yml` and `requirements.txt`.
In the project base folder execute in the command line:

```commandline
conda env create -f environment.yml
pip install -r requirements.txt
```

## Data
From the command line, navigate to the src directory and run: 
```commandline
python get_dataset.py
```

## Running code
From the command line, navigate to the src directory and run: 
```commandline
python preliminary_model.py --study_name {name_of_your_study}
```

## MLflow
Either during or after you have submitted the above model run, open a new terminal <br/>
, navigate to the src directory and run:

```commandline
mlflow ui
```

Follow the link generated to track your model, performance stats, artifacts and piplines.

## Optuna
Similar to the above, open a new terminal <br/>
, navigate to the src directory and run:

```commandline
optuna-dashboard sqlite:///db.sqlite3
```

Follow the link generated to track your study.
