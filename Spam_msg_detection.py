import pandas as pd
import numpy as np


dataset=pd.read_csv("spam\spam.csv",encoding='latin-1')
dataset.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)


import re
import nltk
import joblib

#nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


c=[]



for i in range(0,5572):
    review=re.sub('[^a-zA-Z]',' ',dataset['v2'][i])
    review= review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    c.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1200)   #max_feature can be of any size depending upon size of dataset.
x=cv.fit_transform(c).toarray()


joblib.dump(cv.vocabulary_,"features001.save")    


y=dataset.iloc[:,0].values 

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y=lb.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()


model.add(Dense(input_dim=1200,kernel_initializer='random_uniform',activation='sigmoid',units=1500))
model.add(Dense(units=150,kernel_initializer='random_uniform',activation='sigmoid'))
model.add(Dense(units=1,kernel_initializer='random_uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=50,batch_size=20)

model.save('amodelfile.h5')

y_pred=model.predict(x_test)
y_pred=(y_pred>=0.5)

loaded=CountVectorizer(decode_error='replace',vocabulary=joblib.load('features001.save'))


