import os
import sys

path = "../data"
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print(f"Creating {path} and saving data!")
else:
   print(f"Downloading to {path} directory")

os.system("kaggle datasets download -d dell4010/wine-dataset -p ../data/")