import numpy
import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

cols = ['crim','zn','indus', 'chas', 'nox', 'rm','age','dis','rad','tax','ptratio','black','lstat']
df1 = pd.DataFrame(x_train,columns = cols)
df1['medv'] = y_train
df2 = pd.DataFrame(x_test,columns = cols)
df2['medv'] = y_test
df = pd.concat([df1,df2])
df.to_csv('boston_housing.csv',index=False)