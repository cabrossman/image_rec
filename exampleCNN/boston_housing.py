import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.datasets import boston_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import KFold,cross_val_score

%matplotlib inline 
#prep data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

cols = ['crim','zn','indus', 'chas', 'nox', 'rm','age','dis','rad','tax','ptratio','black','lstat']
df1 = pd.DataFrame(x_train,columns = cols)
df1['medv'] = y_train
df2 = pd.DataFrame(x_test,columns = cols)
df2['medv'] = y_test
df = pd.concat([df1,df2])
df.to_csv('boston_housing.csv',index=False)


model_eval = {}
#RMSE of using training average is =9.188011545278206
avg_rmse = df.medv.apply(lambda x: (df1.medv.mean() - x)**2).mean()**.5

#linear regression
lr = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=None)
lr.fit(X = x_train, y = y_train)
lr_housing_pred = lr.predict(x_test)
lr_rmse = mean_squared_error(y_test, lr_housing_pred)**.5

def k_model():
	model = Sequential()
	model.add(Dense(32, activation = 'relu', kernel_initializer = 'glorot_normal'))
	model.add(Dense(1, activation = 'relu', kernel_initializer = 'glorot_normal'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
model = k_model()
model.fit(x = x_train, y = y_train, epochs=50, batch_size=32)
nn_housing_pred = model.predict(x_test).flatten()
nn_rmse = mean_squared_error(y_test, nn_housing_pred)**.5



estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=k_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
pipeline.fit(X = x_train, y = y_train)
nn_housing_pred_pipeline = pipeline.predict(x_test)
nn_rmse_pipeline = mean_squared_error(y_test, nn_housing_pred_pipeline)**.5

{'avg' : avg_rmse, 'lr_rmse' : lr_rmse, 'nn_rmse' : nn_rmse, 'nn_pipe_rmse' : nn_rmse_pipeline}