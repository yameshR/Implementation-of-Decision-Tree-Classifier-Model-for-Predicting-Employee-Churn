# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: ATHMAJ VENUGOPAL
 
RegisterNumber:  212222240014

```python
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data.head():
![272776023-9d2d241f-6815-4c82-a8f9-a4f98b716269](https://github.com/S-ARVIND01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707337/f14908fe-752c-4cdc-8d5e-07ec7b8602d9)

### Data.info():
![272776163-7f9ac45a-e910-4aec-a636-8c830dd6f052](https://github.com/S-ARVIND01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707337/f73873a4-8ff5-46a9-98c9-faa4be31f6db)

### Isnull() and Sum():
![272776227-067d9b0c-721b-48ae-83c6-cdb3d66866ca](https://github.com/S-ARVIND01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707337/41deaa87-279a-4d7e-ae21-cd605e809c8b)

### DataValue Counts():
![272777333-49142d5a-ae0f-4421-be20-0a40656968c0](https://github.com/S-ARVIND01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707337/8658145d-6bf3-4384-8813-15d8f276bbfd)

### Data.head() for salary:
![272777420-cdacd324-81f1-4bf2-a6f8-549641bc35d7](https://github.com/S-ARVIND01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707337/6454f06a-df7c-4ab8-9c43-0bfac570232a)

### X.Head():
![272777475-196f2e85-bd4c-4e75-8512-6d02618cefa6](https://github.com/S-ARVIND01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707337/093269cf-803a-4df1-b5ab-ba64aa679a21)

### Accuracy Value:
![272777484-d70a371b-840c-41db-9f2b-c6d99b2ca673](https://github.com/S-ARVIND01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707337/da48eaa3-b021-4458-8c1c-9bb4a1bef9cc)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
