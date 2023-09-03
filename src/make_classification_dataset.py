import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot


# Features have already been pre-processed
train_x = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_train_x.csv')
test_x = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_test_x.csv')

# Labels
train_y = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/train_y.csv')
test_y = pd.read_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/test_y.csv')

# Re-balance the labels train
train_y['class'] = 0
train_y.loc[train_y['quality'] > 5, 'class'] = 1
train_x['labels'] = train_y['class']
train_x_bad = train_x[train_x['labels'] == 0]
train_x_good = train_x[train_x['labels'] == 1]
train_x_good_sample = train_x_good.sample(n=np.shape(train_x_bad)[0])

# Re-balance the labels test
test_y['class'] = 0
test_y.loc[test_y['quality'] > 5, 'class'] = 1
test_x['labels'] = test_y['class']
test_x_bad = test_x[test_x['labels'] == 0]
test_x_good = test_x[test_x['labels'] == 1]
test_x_good_sample = test_x_good.sample(n=np.shape(test_x_bad)[0])

# Make the dataset balanced - train
train_x = pd.concat([train_x_good_sample, train_x_bad], axis=0)
train_y = train_x.loc[:, 'labels']
# train_ohe_y = one_hot(torch.tensor(train_y.values))
train_x = train_x.iloc[:, :-1]

# Make the dataset balanced - test
test_x = pd.concat([test_x_good_sample, test_x_bad], axis=0)
test_y = test_x.loc[:, 'labels']
# test_ohe_y = one_hot(torch.tensor(test_y.values))
# test_ohe_y = one_hot(torch.tensor(test_y.values))
test_x = test_x.iloc[:, :-1]

train_x.to_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_classification_train_x.csv', index=False)
test_x.to_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_classification_test_x.csv', index=False)

train_y.to_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_classification_train_y.csv', index=False)
test_y.to_csv('/Users/sum02dean/projects/wine_challenge/WINE/data/scaled_unclipped_classification_test_y.csv', index=False)
