# EX 09: Implementation-of-SVM-For-Spam-Mail-Detection

## DATE:

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Character Encoding Detection:

* You’ve used the chardet library to detect the character encoding of the file “spam.csv”.

* The detected encoding is likely to be Windows-1252 (also known as cp1252).
Data Loading and Exploration:

* You’ve loaded the dataset from “spam.csv” using pandas and specified the encoding as Windows-1252.

* You’ve printed the first five rows of the dataset using data.head().

* Additionally, you’ve displayed information about the dataset using data.info().

2. Data Preprocessing:

* You’ve split the data into training and testing sets using train_test_split.

* You’ve used CountVectorizer to convert text data (in column “v2”) into numerical features for SVM training.

3. Model Training and Prediction:

* You’ve initialized an SVM classifier (svc) and trained it on the training data.

* You’ve predicted the labels for the test data using y_pred.

4. Model Evaluation:

* You’ve calculated the accuracy of the model using metrics.accuracy_score.

## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VENKATANATHAN P R
RegisterNumber:  212223240173
*/
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
print("The First five Data:\n")
print(data.head())
print("\nThe Information:\n")
print(data.info())
print("\nTo count the Number of null values in dataset:\n")
print(data.isnull().sum())
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print("\nThe Y_prediction\n")
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("\nAccuracy:\n")
print(accuracy)

```

## Output:

![alt text](<Screenshot 2024-05-10 234159.png>)

![alt text](<Screenshot 2024-05-10 234206.png>)

![alt text](<Screenshot 2024-05-10 234213.png>)

![alt text](<Screenshot 2024-05-10 234221.png>)

![alt text](<Screenshot 2024-05-10 234225.png>)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
