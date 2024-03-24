import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('./train.csv')
print(data.head(10))
print(data.isnull().sum())
#null values are Age, Cabin, and Embarked.
print(data.columns)
# f, ax = plt.subplots(1,2, figsize = (18,8))
# a = data['Survived'].value_counts().plot.pie(autopct= '%1.1f%%', ax =ax[0], explode=[0,0.1], shadow= True)
# x = ax[0].set_title('Survived')
# y = ax[0].set_ylabel('')
# m = sns.countplot(x = 'Survived', data = data , ax = ax[1])
# p = ax[1].set_title('Survived')
# plt.show()
# out of 891 passengers around 350 survived and rest are dead.

# x = data.groupby(['Survived','Sex'])['Survived'].count()
# print(x)
# f, ax = plt.subplots(1,2,figsize = (18,8))
# y = data[['Survived','Sex']].groupby(['Sex'])['Survived'].mean().plot.bar(ax=ax[0])
# m = ax[0].set_title('Survived vs Sex')
# n = sns.countplot(x = 'Sex', data = data ,hue= 'Survived', ax= ax[1])
# o = ax[1].set_title('sex: Survived vs Dead')
# plt.show()
#women survived more than men

# x = data.groupby(['Survived','Pclass'])['Survived'].count()
# print(x)
# f, ax = plt.subplots(1,2,figsize= (18,8))
# o = data[['Survived','Pclass']].groupby(['Pclass'])['Survived'].mean().plot.bar(ax = ax[0], color = ['#CD7F32', '#FFDF00','#D3D3D3'])
# k = ax[0].set_title('Pclass vs Survived')
# y = sns.countplot(x = 'Pclass', hue = 'Survived', data = data, ax = ax[1])
# d = ax[1].set_title('Pclass : Survived vs Dead')
# plt.show()
# people of Pclass 1 had highest survival rate as compared to Pclass 2 and 3.

# x = data.groupby(['Sex','Pclass'])['Sex'].count()
# print(x)
# f, ax = plt.subplots(1,4,figsize= (18,8))
# u = data[['Sex','Pclass']].groupby(['Pclass'])['Sex'].count().plot.bar(color = ['#CD7F32', '#FFDF00','#D3D3D3'] , ax = ax[0])
# y = ax[0].set_title('Pclass vs number of people')
# m = sns.barplot(x = 'Sex', y =data['Pclass']== 1, data = data, hue ='Survived', ax =ax[1])
# t = ax[1].set_title('Pclass 1 : Sex vs Survived')
# m = sns.barplot(x = 'Sex', y =data['Pclass']== 2, data = data, hue ='Survived', ax =ax[2])
# t = ax[2].set_title('Pclass 2 : Sex vs Survived')
# m = sns.barplot(x = 'Sex', y =data['Pclass']== 3, data = data, hue ='Survived', ax =ax[3])
# t = ax[3].set_title('Pclass 3 : Sex vs Survived')
# plt.show()


# x = pd.crosstab([data.Sex,data.Survived], [data.Pclass], margins = True)
# display(x)
# y = data.groupby(['Sex','Pclass','Survived'])['Survived'].count()
# print(y)

# z = sns.catplot(x ='Pclass',y = 'Survived',hue = 'Sex', data = data, kind = 'point')
# plt.show()
# it is evident that irrespective of Pclass, women were given first priority while rescue.


print(data['Age'].describe())
#oldest passenger was of 80 years
#youngest passenger was of 0.42 years
#Average age on the ship was 29.699118 years

# f, ax = plt.subplots(1,2, figsize =(18,8))
# e = sns.violinplot(x = 'Pclass', y = 'Age', hue ='Survived', data = data , split= True, ax =ax[0] )
# t = ax[0].set_title('Pclass and Age vs Survived')
# y = ax[0].set_yticks(range(0,110,10))
# u = sns.violinplot(x = 'Sex', y = 'Age', hue = 'Survived', data = data, ax = ax[1], split = True)
# h = ax[1].set_title('Sex and Age vs Survived')
# d = ax[1].set_yticks(range(0,110,10))
# plt.show()

#the number of children increases with Pclass and the survival rate for passengers below age 10 looks to be good irrespective of the Pclass.
#Survival chances for passengers aged 20-50 from Pclass 1 is high and is even better for women.
#for males, the survival chances decreases with an increase in age.


print(data['Age'].isnull().sum()) #177 null values for Age
x = data['Initial'] = 0
for i in data :
    data['Initial']= data.Name.str.extract('([A-Za-z]+)\.') #to extract the salutations

p = pd.crosstab([data.Initial], [data.Sex])
print(p)
y = data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Col','Capt','Don', 'Jonkheer','Rev','Sir'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Mr','Mr','Other','Other','Mr'], inplace = True)

print(data.groupby(['Initial'])['Age'].mean())
# Master     4.574167
# Miss      21.860000
# Mr        32.739609
# Mrs       35.981818
# Other     45.888889

t = data.loc[(data.Age.isnull()) & (data.Initial == 'Mr'), 'Age'] = 33
u = data.loc[(data.Age.isnull()) & (data.Initial == 'Miss'), 'Age'] = 22
e = data.loc[(data.Age.isnull()) & (data.Initial == 'Master'), 'Age'] = 5
r = data.loc[(data.Age.isnull()) & (data.Initial == 'Mrs'), 'Age'] = 36
w = data.loc[(data.Age.isnull()) & (data.Initial == 'Other'), 'Age'] = 46
print(data.Age.isnull().sum()) # all the null values for age are being filled.

# f, ax = plt.subplots(1,2,figsize =(20,10))
# ay = data[data['Survived']==0].Age.plot.hist(ax = ax[0], bins = 20, edgecolor='black', color = 'red' )
# et = ax[0].set_title('Survived = 0')
# x1 = list (range(0,85,5))
# x2 = ax[0].set_xticks(x1)
# x3 = data[data['Survived']==1].Age.plot.hist(ax = ax[1], color = 'green', bins = 20, edgecolor= 'black')
# x4 = ax[1].set_title('Survived = 1')
# x6 = list (range(0,85,5))
# x5 = ax[1].set_xticks(x6)
# plt.show()

#the oldest passenger was saved (80 years)
#women and child first policy
#maximum number of deaths were in the age group of 30-40.

# x7 = sns.catplot(x ='Pclass', y = 'Survived', col = 'Initial', data = data)
# plt.show()

#the women and child first policy thus holds true irrespective of the class.
x1 = pd.crosstab([data.Embarked,data.Pclass], [data.Sex,data.Survived],margins = True)
print(x1)
# x2 = sns.catplot(x = 'Embarked', y = 'Survived', data = data)
# fig = plt.gcf()
# x3 = fig.set_size_inches(5,3)
# plt.show()
# The chances for survival for port C is highest around 0.55 while it is lowest for S.

# f, ax = plt.subplots(2,2, figsize = (20,15))
# x2 = sns.countplot(x = 'Embarked', data = data, ax = ax[0,0])
# x3 = ax[0,0].set_title('No. of Passengers Boarded')
# x4 = sns.countplot(x = 'Embarked', hue = 'Sex', ax = ax[0,1], data = data)
# x5 = ax[0,1].set_title('Male-Female Split for Embarked')
# x6 = sns.countplot(x= 'Embarked',hue ='Survived', ax = ax[1,0], data = data)
# x7 = ax[1,0].set_title('Embarked vs Survived')
# x8 = sns.countplot(x= 'Embarked',hue ='Pclass', ax = ax[1,1], data = data)
# x9 = ax[1,1].set_title('Embarked vs Pclass')
# x10 = plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
# plt.show()


#maximum passengers boarded from port S followed by ports C and Q same for sex in ports
#most number of deaths were from port S and most number of survived are also from port S
#most of the passengers from Pclass 3 were boarded from port S
#The Embark S looks to the port from where majority of the rich people boarded. Still the chances for survival is low here, that is because many passengers from Pclass3 around 81% didn't survive.
#port Q has majority Pclass 3 people

# x2 = sns.catplot(x = 'Pclass', y = 'Survived', hue = 'Sex', col = 'Embarked', data = data, kind = "point")
# plt.show()


#filling Embarked NaN values (number of null values = 2)
print(data.Embarked.isnull().sum())
x2 = data.Embarked.fillna('S',inplace = True) #because many passengers were from S port
print(data.Embarked.isnull().any())

#SibSip (a discrete feature represents whether a person is alone or with his family members.
x3 = pd.crosstab([data.SibSp],[data.Survived], margins = True)
print(x3)
#people who were alone had least survival chances.
#people with more than 5 family members also had very less survival chances.
# f, ax = plt.subplots(1,2,figsize=(18,8))
# y2 = data[['SibSp','Survived']].groupby(['SibSp'])['Survived'].mean().plot.bar(ax= ax[0])
# y1 = sns.catplot(x = 'SibSp', y = 'Survived', data = data, ax = ax[1])
# plt.show()

y3 = pd.crosstab([data.SibSp],[data.Pclass], margins = True)
print(y3)
#people who were single or have a partner were majority from P class 1
#people having high SibSp value (bigger family) travelled in Pclass 3 thats why their survival rate was low

y4 = pd.crosstab([data.Parch], [data.Pclass], margins = True)
print(y4)
#larger families were in Pclass 3

y5 = data.groupby(['Parch','Survived'])['Survived'].count()
print(y5)

# f, ax = plt.subplots(1,2,figsize = (18,8))
# y6 = data[['Parch','Survived']].groupby(['Parch'])['Survived'].mean().plot.bar(ax= ax[0])
# y7 = ax[0].set_title('Parch vs Survived')
# y8 = sns.catplot(x = 'Parch', y = 'Survived', data = data , ax = ax[1])
# y9 = ax[1].set_title('Parch vs Survived')
# plt.show()

#the chances for the survival is good for soembody who has 1-3 parents on the ship. being alone  also proves to be fatal and the chances for survival decreases when soembody has >4 parents on the ship.

print(data.Fare.describe())
#highest fare was 512.3292
#lowest fare was 0 (free luxury ride)
#mean fare was 32.204208

# f, ax = plt.subplots(1,3,figsize= (18,8))
# y6 = sns.distplot(data[data['Pclass']==1].Fare,ax = ax[0])
# y7 = ax[0].set_title('Fares in Pclass 1')
# y8 = sns.distplot(data[data['Pclass']==2].Fare, ax = ax[1])
# y9 = ax[1].set_title('Fares in Pclass 2')
# y10 = sns.distplot(data[data['Pclass']==3].Fare,ax = ax[2])
# y11 = ax[2].set_title('Fares in Pclass 3')
# plt.show()
# numeric_data = data.select_dtypes(include = ['number'])
# y6 = sns.heatmap(numeric_data.corr(), annot=True, linewidths = 0.2, cmap ='RdYlGn')
# fig = plt.gcf()
# fig.set_size_inches(10,8)
# plt.show()

print(data.Age.describe())
z1 = data['Age_band']=0
y6 = data.loc[(data['Age']<=16), 'Age_band']= 0
y7 = data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
y8 = data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
y9 = data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
y6 = data.loc[(data['Age']>64), 'Age_band']= 4
print(data.head(2))

z2 = data.groupby(['Age_band'])['Age_band'].count()
print(z2)
# z3 = sns.countplot(x = 'Age_band', data = data, hue = 'Sex')
# plt.show()
# z4 = sns.catplot(x ='Age_band',y = 'Survived',data = data, col= 'Pclass')
# plt.show()
#thus the survival rate decreases as the age increases irrespective of the Pclass.

z3 = data['Family_size']= 0
data['Family_size'] = data['Parch'] + data['SibSp'] #family size
data['Alone']= 0
z4 = data.loc[(data['Family_size'] == 0), 'Alone']=1

# f, ax = plt.subplots(1,2,figsize= (18,6))
# z5 = data[['Family_size','Survived']].groupby(['Family_size'])['Survived'].mean().plot.bar(ax = ax[0])
# z6 = sns.barplot(x = 'Alone', y = 'Survived', data = data, ax = ax[1])
# z7 = ax[0].set_title('Family_size vs Survived')
# z8 = ax[1].set_title('Alone vs Survived')
# plt.show()
#peopple who were not alone had high survival chances.


# z5 = sns.barplot(x = 'Alone', y = 'Survived',hue = 'Pclass' ,data = data)
# plt.show()
#people having family have high survival rate irrespective of Pclass

z5 = data['Fare_Range'] = pd.qcut(data['Fare'],4)
z6 = data.groupby(['Fare_Range'])['Survived'].mean().to_frame()
print(z6)
# as the fare_range increases, the chances for survival also increases.

z7 = data['Sex'].replace(['male','female'],['0','1'],inplace = True)
z8 = data['Embarked'].replace(['S','C','Q'],['0','1','2'], inplace = True)
z9 = data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],['0','1','2','3','4'], inplace = True)

z10 = data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'], axis = 1, inplace = True)
# z11 = sns.heatmap(data.corr(), annot = True, cmap ='RdYlGn', linewidths = 0.2, annot_kws = {'size':20})
# fig = plt.gcf()
# fig.set_size_inches(18,15)
# plt.xticks(fontsize = 14)
# plt.yticks(fontsize = 14)
# plt.show()
print(data.info())
print(data.head())
print(data.columns)
f,ax = plt.subplots(1,2,figsize = (18,8))
plt1 = sns.heatmap(data.corr(),cmap = 'RdYlGn',annot=True,annot_kws ={'size':20},linewidths=0.2,ax = ax[0])
plt2 = sns.catplot(data = data, kind='point',x='Pclass',y='Survived' ,ax = ax[1],hue = 'Sex')
plt.show()

#Predictive Modelling
import random
from sklearn.model_selection import train_test_split
random.seed(42)
train_df,val_df = train_test_split(data,test_size=0.2,random_state=42)
print(train_df.head())

input_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Initial',
       'Age_band', 'Family_size', 'Alone']
target_cols = ['Survived']
train_inputs = train_df[input_cols]
train_targets = train_df[target_cols]
val_inputs = val_df[input_cols]
val_targets = val_df[target_cols]
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(train_inputs,train_targets)
pred1 = model1.predict(train_inputs)
pred2 = model1.predict(val_inputs)
a1 = accuracy_score(pred1,train_targets)
a2 = accuracy_score(pred2,val_targets)
print(a1)
print(a2)
# 0.8146067415730337
# 0.8044692737430168

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(max_depth=5,max_leaf_nodes=32)
model2.fit(train_inputs,train_targets)
pred3 = model2.predict(train_inputs)
pred4 = model2.predict(val_inputs)
a3 = accuracy_score(pred3,train_targets)
a4 = accuracy_score(pred4,val_targets)
print(a3)
print(a4)
# 0.8384831460674157
# 0.8268156424581006

from sklearn.ensemble import HistGradientBoostingClassifier
model3 = HistGradientBoostingClassifier(max_depth=8,max_iter=100)
model3.fit(train_inputs,train_targets)
pred5 = model3.predict(train_inputs)
pred6 = model3.predict(val_inputs)
a5 = accuracy_score(pred5,train_targets)
a6 = accuracy_score(pred6,val_targets)
print(a5,a6)
# 0.8567415730337079 0.8268156424581006

from sklearn.svm import SVC
model4 = SVC()
model4.fit(train_inputs,train_targets)
pred7 = model4.predict(train_inputs)
pred8 = model4.predict(val_inputs)
a7 = accuracy_score(pred7,train_targets)
a8 = accuracy_score(pred8,val_targets)
print(a7,a8)

from sklearn.neighbors import KNeighborsClassifier
model5 = KNeighborsClassifier()
model5.fit(train_inputs,train_targets)
pred9 = model5.predict(train_inputs)
pred10 = model5.predict(val_inputs)
a9 = accuracy_score(pred9,train_targets)
a10 = accuracy_score(pred10,val_targets)
print(a9,a10)

train_inputs = pd.DataFrame(train_inputs)
train_inputs['Sex']= train_inputs['Sex'].astype(int)
train_inputs['Embarked']= train_inputs['Embarked'].astype(int)
train_inputs['Initial']= train_inputs['Initial'].astype(int)
val_inputs['Sex']= val_inputs['Sex'].astype(int)
val_inputs['Embarked']= val_inputs['Embarked'].astype(int)
val_inputs['Initial']= val_inputs['Initial'].astype(int)

import xgboost as xgb
model7 = xgb.XGBClassifier()
model7.fit(train_inputs,train_targets)
pred13 = model7.predict(train_inputs)
pred14 = model7.predict(val_inputs)
a13 = accuracy_score(pred13,train_targets)
a14 = accuracy_score(pred14,val_targets)
print(a13,a14)


from sklearn.ensemble import VotingClassifier
model6 = VotingClassifier(estimators = [('model2',model2),('model3',model3),('model7',model7)])
model6.fit(train_inputs,train_targets)
pred11=  model6.predict(train_inputs)
pred12 = model6.predict(val_inputs)
a11 = accuracy_score(pred11,train_targets)
a12 = accuracy_score(pred12,val_targets)
print(a11,a12)
# 0.8581460674157303 0.8324022346368715     (maximum accuracy reached)