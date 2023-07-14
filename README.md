# WINE
Wine classification project
![APM](https://img.shields.io/apm/l/vim-mode) 
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg?style=flat)](https://github.com/feross/standard)
<img src="../dessert-wines-what-to-know-FT-BLOG0122-3b0040474802467bb57ea6ee5e4acd92.jpgs" width="450">

## Setup
Dependencies for conda and pip are listed in `environment.yml` and `requirements.txt`.
In the project base folder execute in the command line:

```commandline
conda env create -f environment.yml
pip install -r requirements.txt
```

## Data
From the command line, navigate to the src directory and run: 
```python
python get_dataset.py
```