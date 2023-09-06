#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("homicide-data.csv", encoding='iso-8859-1')


# In[3]:


df.head(4)


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df=df.dropna()


# In[7]:


df.isnull().sum()


# In[8]:


df['disposition'].value_counts()


# In[9]:


df.drop(columns = 'uid' , inplace = True)


# In[10]:


df.drop(columns = 'victim_last' , inplace = True)
df.drop(columns = 'victim_first' , inplace = True)
df.drop(columns = 'state' , inplace = True)


# In[11]:


df.head(5)


# In[12]:


df.drop(columns = 'reported_date' , inplace = True)
df.drop(columns = 'city' , inplace = True)


# In[13]:


df.head(5)


# In[14]:


status_mapping = {
    'Closed by arrest': 0,
    'Open/No arrest': 1,
    'Closed without arrest': 2
}

df['disposition_int'] = df['disposition'].map(status_mapping)


# In[15]:


df['victim_sex'].value_counts()


# In[16]:


status_mapping_sex = {
    'Female': 0,
    'Male': 1,
    'Unknown': 2
}

df['Victim_Sex_int'] = df['victim_sex'].map(status_mapping_sex)


# In[17]:


df['victim_race'].value_counts()


# In[18]:


status_mapping_race = {
    'Black': 0,
    'Hispanic': 1,
    'White': 2,
    'Unknown' : 3,
    'Other' : 4,
    'Asian' :5
}

df['Victim_Race_int'] = df['victim_race'].map(status_mapping_race)


# In[19]:


df.head(10)


# In[20]:


df.drop(columns = 'victim_race' , inplace = True)
df.drop(columns = 'victim_sex' , inplace = True)
df.drop(columns = 'disposition' , inplace = True)


# In[21]:


df.head(10)


# In[22]:


df.sample(10)


# In[23]:


df['victim_age'].value_counts()


# In[24]:


df.replace('Unknown', 0, inplace=True)


# In[25]:


df['victim_age'].value_counts()


# In[27]:


X = df.drop('disposition_int', axis=1)
y = df['disposition_int']


# In[28]:


X


# In[29]:


y


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[45]:


model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)
# Calculate the R-squared score
r_squared = r2_score(y_test, y_pred)
print(f'R-squared Score: {r_squared}')
print(y_pred[:5])
print(y_test[:5])


# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression(random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print("Accuracy:", accuracy)
print(y_pred[:5])
print(y_test[:5])


# In[42]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print("Accuracy:", accuracy)
print(y_pred[:5])
print(y_test[:5])


# In[41]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(y_pred[:5])
print(y_test[:5])


# In[44]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(y_pred[:5])
print(y_test[:5])


# In[46]:


from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


# In[47]:


regressor = DecisionTreeRegressor(random_state=42)

# Fit the model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Calculate the R-squared score
r_squared = r2_score(y_test, y_pred)

print(f"R-squared score: {r_squared}")
print(y_pred[:5])
print(y_test[:5])

