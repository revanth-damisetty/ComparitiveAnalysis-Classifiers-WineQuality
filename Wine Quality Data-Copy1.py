# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn .metrics import accuracy_score,roc_auc_score,roc_curve


# In[2]:


wine = pd.read_csv('winequality-red.csv')
print("Sample of the Dataset:\n")
print(wine.head())
print("\n")
print("Stastical Description of the DataSet:")
print(wine.describe())



# In[6]:


fig = plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
sns.barplot(x='quality',y='fixed acidity',data=wine)

plt.subplot(2,3,2)
sns.barplot(x='quality',y='volatile acidity',data=wine)

plt.subplot(2,3,3)
sns.barplot(x='quality',y='citric acid',data=wine)

plt.subplot(2,3,4)
sns.barplot(x='quality',y='residual sugar',data=wine)

plt.subplot(2,3,5)
sns.barplot(x='quality',y='chlorides',data=wine)

plt.subplot(2,3,6)
sns.barplot(x='quality',y='free sulfur dioxide',data=wine)

plt.tight_layout()

plt.show()

fig = plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
sns.barplot(x='quality',y='total sulfur dioxide',data=wine)

plt.subplot(2,3,2)
sns.barplot(x='quality',y='density',data=wine)

plt.subplot(2,3,3)
sns.barplot(x='quality',y='pH',data=wine)

plt.subplot(2,3,4)
sns.barplot(x='quality',y='sulphates',data=wine)

plt.subplot(2,3,5)
sns.barplot(x='quality',y='alcohol',data=wine)

plt.tight_layout()
plt.show()



# In[7]:


#from 2 to 6.5 it is considered bad and above that it is good as 8 is the max value of quality
ranges = (2,6,8) 
groups = ['bad','good']
wine['quality'] = pd.cut(wine['quality'],bins=ranges,labels=groups)

le = LabelEncoder()
wine['quality'] = le.fit_transform(wine['quality'])

good_quality = wine[wine['quality']==1]
bad_quality = wine[wine['quality']==0]

bad_quality = bad_quality.sample(frac=1)
bad_quality = bad_quality[:len(good_quality)]

new_df = pd.concat([good_quality,bad_quality])
new_df = new_df.sample(frac=1)

new_df.corr()['quality'].sort_values(ascending=False)


# In[13]:

def report(y_test,pred,Algo):
    print("\nReport for",Algo,":\n")
    print("-->Confusion Matrix:")
    print(confusion_matrix(y_test,pred))
    print('\n')
    print("-->Classification Report:")
    print(classification_report(y_test,pred))
    print('\n')
    print("-->Accuracy Score:",accuracy_score(y_test,pred))
    print('\n')
    fpr, tpr, _ = roc_curve(y_test, pred)
    auc = round(roc_auc_score(y_test, pred), 4)
    print(Algo+"'s AUC Score is "+str(auc)+"\n")
    plt.plot(fpr,tpr,label=Algo+", AUC="+str(auc))
    
    


X = new_df.drop('quality',axis=1) 
y = new_df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[14]:
    
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn = knn.predict(X_test)
auc = np.round(roc_auc_score(y_test, knn), 3)

report(y_test, knn, "K Nearest Neighbour Classifier")

#Support Vector Machine:

svm = SVC(kernel='rbf', random_state = 101)
svm.fit(X_train,y_train)
svm = svm.predict(X_test)

report(y_test,svm,"Support Vector Machine")

#Decision Tree: 

gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)
gini.fit(X_train, y_train)
gini = gini.predict(X_test)

report(y_test,gini,"Decision Tree Using Gini Index")

entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5)
entropy.fit(X_train, y_train)
entropy = entropy.predict(X_test)

report(y_test,entropy,"Decision Tree Using Entropy")

#Random Forest Classifier

print("Random Forest Classifier:\n")

param = {'n_estimators':[100,200,300,400,500,600,700,800,900,1000]}

rfc = GridSearchCV(RandomForestClassifier(),param,scoring='accuracy',cv=10,)
rfc.fit(X_train, y_train)

print('Best parameters --> ', rfc.best_params_)
print("\n")

rfc = rfc.predict(X_test)

report(y_test,rfc,"Random Forest Classsifier")

arr=np.arange(0,1.1,0.1)
plt.plot(arr,arr,linestyle='dashed')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Reciever Operating Characteristics")
    

plt.legend()




 




