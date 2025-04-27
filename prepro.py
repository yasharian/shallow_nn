import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
test_data = pd.read_csv("./data/diabetes_test.csv")
test_data.head()
train_data_outcome = train_data['Outcome']
train_data = train_data.drop(columns = 'Outcome'  , inplace = False ) # TODO: drop Outcome

train_data.head()
for column in train_data.columns:
  mean = train_data[column].mean() # TODO (Mean of train_data[column])
  std = train_data[column].std()  # TODO (Standard Deviation of train_data[column])
  train_data[column]=(train_data[column]- mean)/std
  test_data[column]=(test_data[column] - mean)/std
    
train_data['Bias'] = 1  
train_bias =  train_data['Bias'] # TODO

test_data['Bias'] = 1  # TODO: add Bias to test_data
test_bias = test_data['Bias'] # TODO

train_data.head()

X_train, X_validation, y_train, y_validation = train_test_split(train_data, train_data_outcome, test_size=0.2) 

X_train = np.array(X_train).T # TODO
X_validation = np.array(X_validation) # TODO
y_train = np.array(y_train).T # TODO
y_validation = np.array(y_validation) # TODO
test_data_numpy = test_data.to_numpy(dtype = 'float32').T # TODO

print(f'X_train.shape:{X_train.shape}, y_train.shape:{y_train.shape}')
print(f'X_validation.shape:{X_validation.shape}, y_validation.shape:{y_validation.shape}')
print(f'test_data_numpy.shape:{test_data_numpy.shape}')