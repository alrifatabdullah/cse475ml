# -*- coding: utf-8 -*-


import pandas as pd
mydataset = pd.read_csv('dataset_dep.csv')

x = mydataset.iloc[:,3:6].values
y = mydataset.iloc[:,2].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_1.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_feature = [1])
x= onehotencoder.fit_transform(x).toarray()
x=x[:,1:]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)





# i_train = pd.DataFrame(x_train)
# i_test = pd.DataFrame(x_test)
# j_train = pd.DataFrame(y_train)
# j_test = pd.DataFrame(y_test)



