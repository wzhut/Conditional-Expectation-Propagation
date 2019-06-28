#%%
# ionos
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp
from sklearn.utils import shuffle
import math

# load data
ionos = pd.read_csv('./ionosphere.data.txt', header=None)
# normalization
standard_scaler = pp.StandardScaler()
x = ionos.iloc[:,0:34].values.astype(float)
x_scaled = standard_scaler.fit_transform(x)
ionos.iloc[:,0:34] = x_scaled
# convert categorical data into numerical data
replace={34: {'g':1, 'b':0}}
ionos.replace(replace, inplace=True)

nfold = 5
num_train = math.ceil(ionos.shape[0] / 2)
for i in range(1, nfold+1):
    data = shuffle(ionos)
    train = pd.DataFrame(data = data.iloc[:num_train,:], index=None)
    test = pd.DataFrame(data = data.iloc[num_train:,:], index=None)
    train.to_csv('./ionos/train-'+str(i)+'.csv', header=None, index=False)
    test.to_csv('./ionos/test-'+str(i)+'.csv', header=None, index=False)
ionos.to_csv('ionos.csv', header=None, index=False)

#%%
# breast
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp

# load data
breast = pd.read_csv('./breast-cancer.data.txt', header=None)
breast = breast[(breast[[5,8]] != '?').all(axis=1)]
breast = breast[[1,2,3,4,5,6,7,8,9,0]]

out = pd.DataFrame(index=None)
## convert categorical data into binary data
# replace={0: {'no-recurrence-events': 0, 'recurrence-events':1},
# 1: {'10-19':0, '20-29':1, '30-39':2, '40-49':3, '50-59':4, '60-69':5, '70-79':6, '80-89':7, '90-99':8},
# 2: {'lt40':0, 'ge40':1, 'premeno':2},
# 3: {'0-4':0, '5-9':1, '10-14':2, '15-19':3, '20-24':4, '25-29':5, '30-34':6, '35-39':7, '40-44':8, '45-49':9, '50-54':10, '55-59':11},
# 4: {'0-2': 0, '3-5':1, '6-8':2, '9-11':3, '12-14':4, '15-17':6, '18-20':7, '21-23':8, '24-26':9,'27-29':10, '30-32':11, '33-35':12, '36-39':13},
# 5: {'yes': 1, 'no':0},
# 7: {'left': 0, 'right': 1},
# 8: {'left_up': 0, 'left_low':1, 'right_up':2, 'right_low':3, 'central':4},
# 9: {'yes':1, 'no': 0}
# }
cc = 0
for c in breast.columns:
    uv = sorted(breast[c].unique())
    # print(uv)
    if len(uv) > 2:
        for v in uv:
            # print(v)
            out[cc] = breast[c].apply(lambda x: 1 if x == v else 0)
            cc+=1
    else:
        out[cc] = breast[c].apply(lambda x: 1 if x == uv[0] else 0)
        cc+=1

replace = {cc-1: {-1:0}}
out.replace(replace, inplace=True)
out.to_csv('breast.csv', header=None, index=False)

nfold = 5
num_train = math.ceil(out.shape[0] / 2)
for i in range(1, nfold+1):
    data = shuffle(out)
    train = pd.DataFrame(data = data.iloc[:num_train,:], index=None)
    test = pd.DataFrame(data = data.iloc[num_train:,:], index=None)
    train.to_csv('./breast/train-'+str(i)+'.csv', header=None, index=False)
    test.to_csv('./breast/test-'+str(i)+'.csv', header=None, index=False)

#%%
# sonar
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp

# load data
sonar = pd.read_csv('./sonar.all-data.txt', header=None)
# normalization
standard_scaler = pp.StandardScaler()
x = sonar.iloc[:,0:60].values.astype(float)

x_scaled = standard_scaler.fit_transform(x)
sonar.iloc[:,0:60] = x_scaled
# label
replace = {60: {'R':0, 'M':1}}
sonar.replace(replace, inplace=True)
sonar.to_csv('sonar.csv', header=None, index=False)

nfold = 5
num_train = math.ceil(sonar.shape[0] / 2)
for i in range(1, nfold+1):
    data = shuffle(sonar)
    train = pd.DataFrame(data = data.iloc[:num_train,:], index=None)
    test = pd.DataFrame(data = data.iloc[num_train:,:], index=None)
    train.to_csv('./sonar/train-'+str(i)+'.csv', header=None, index=False)
    test.to_csv('./sonar/test-'+str(i)+'.csv', header=None, index=False)



#%%
# australian
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp

aus = pd.read_csv('./australian.dat.txt', header=None, sep=' ')
# print(aus.head())
continuous = [1, 2, 6, 9, 12, 13]
categorical = [0, 3, 4, 5, 7, 8, 10, 11]
# normalization
standard_scaler = pp.StandardScaler()
x = sonar.iloc[:,continuous].values.astype(float)

x_scaled = standard_scaler.fit_transform(x)

out = pd.DataFrame(data = x_scaled, index=None)

# out.iloc[:,0:5] = x_scaled
# print(aus.columns)
cc = 6
for c in categorical:
    uv = sorted(aus[c].unique())
    # print(uv)
    if len(uv) > 2:
        for v in uv:
            # print(v)
            out[cc] = aus[c].apply(lambda x: 1 if x == v else 0)
            cc+=1
    else:
        out[cc] = aus[c].apply(lambda x: 1 if x == uv[0] else 0)
        cc+=1
out[cc] = aus[14]
out.to_csv('./australian.csv', header=None, index=False)

nfold = 5
num_train = math.ceil(out.shape[0] / 2)
for i in range(1, nfold+1):
    data = shuffle(out)
    train = pd.DataFrame(data = data.iloc[:num_train,:], index=None)
    test = pd.DataFrame(data = data.iloc[num_train:,:], index=None)
    train.to_csv('./australian/train-'+str(i)+'.csv', header=None, index=False)
    test.to_csv('./australian/test-'+str(i)+'.csv', header=None, index=False)


#%%
# pima
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp

pima = pd.read_csv('./pima.data.txt', header=None)

# normalization
standard_scaler = pp.StandardScaler()
x = pima.iloc[:,0:8].values.astype(float)
x_scaled = standard_scaler.fit_transform(x)
pima.iloc[:,0:8] = x_scaled

pima.to_csv('./pima.csv', header=None, index=False)

nfold = 5
num_train = math.ceil(pima.shape[0] / 2)
for i in range(1, nfold+1):
    data = shuffle(pima)
    train = pd.DataFrame(data = data.iloc[:num_train,:], index=None)
    test = pd.DataFrame(data = data.iloc[num_train:,:], index=None)
    train.to_csv('./pima/train-'+str(i)+'.csv', header=None, index=False)
    test.to_csv('./pima/test-'+str(i)+'.csv', header=None, index=False)


#%%
# crabs
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp

crabs = pd.read_csv('./crabs.data.txt', header=None)

# normalization
standard_scaler = pp.StandardScaler()
x = crabs.iloc[:,1:6].values.astype(float)
x_scaled = standard_scaler.fit_transform(x)
crabs.iloc[:,1:6] = x_scaled

replace={0: {'B':0, 'O':1}, 6: {'F': 0, 'M':1}}
crabs.replace(replace, inplace=True)

crabs.to_csv('./crabs.csv', header=None, index=False)

nfold = 5
num_train = math.ceil(crabs.shape[0] / 2)
for i in range(1, nfold+1):
    data = shuffle(crabs)
    train = pd.DataFrame(data = data.iloc[:num_train,:], index=None)
    test = pd.DataFrame(data = data.iloc[num_train:,:], index=None)
    train.to_csv('./crabs/train-'+str(i)+'.csv', header=None, index=False)
    test.to_csv('./crabs/test-'+str(i)+'.csv', header=None, index=False)
