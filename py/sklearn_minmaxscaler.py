import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data=np.random.randint(0, 100, (10, 2))

scaler_model=MinMaxScaler()

scaler_model.fit(data)
data=scaler_model.transform(data)

df=pd.DataFrame(data=data, columns=['f1', 'f2', 'f3', 'label'])

X=df[['f1', 'f2', 'f3']]
y=df['label']
X_train, y_train, X_test, y_test=train_test_split(X,y, test_size=0.3, random_state=101)

