#!/usr/bin/env python
# coding: utf-8

# # Phase One : Data set & Data Description

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import f1_score
import missingno as msno


# In[2]:


train_df = pd.read_csv(r"C:\Users\sandr\Desktop\churn-bigml-80.csv")
test_df = pd.read_csv(r"C:\Users\sandr\Desktop\churn-bigml-20.csv")
(train_df.head())


# ### combining training and testing datasets into a single DataFrame

# In[3]:


train_df['Data Set'] = 'train'
test_df['Data Set'] = 'test'

combined_df = pd.concat([train_df, test_df], ignore_index=True)


# In[4]:


(combined_df.info())


# # Phase Two : Exploratory data analysis

# ## Compute descriptive statistics for numerical variables

# In[5]:


combined_df.describe()


# In[6]:


combined_df.head()


# In[7]:


combined_df.nunique()


# ## some Plot histograms for numerical variables

# In[8]:


combined_df.columns = combined_df.columns.str.replace(' ', '_')


# In[9]:


numerical_variables = ['Total_day_minutes', 'Total_day_calls', 'Total_day_charge', 'Total_eve_minutes', 'Total_eve_calls', 'Total_eve_charge', 'Total_night_minutes', 'Total_night_calls', 'Total_night_charge', 'Total_intl_minutes', 'Total_intl_calls', 'Total_intl_charge', 'Customer_service_calls']
combined_df[numerical_variables].hist(figsize=(12, 10))
plt.tight_layout()
plt.show()


# ## bar charts for categorical variables

# In[10]:


categorical_variables = ['Area_code', 'International_plan', 'Voice_mail_plan', 'Churn']
for var in categorical_variables:
    combined_df[var].value_counts().plot(kind='bar')
    plt.title(var)
    plt.show()


# ### Compute correlation matrix for numerical variables in a heatmap

# In[11]:


numerical_columns = ['Account_length', 'Number_vmail_messages', 'Total_day_minutes', 'Total_day_calls', 'Total_day_charge', 'Total_eve_minutes', 'Total_eve_calls', 'Total_eve_charge', 'Total_night_minutes', 'Total_night_calls', 'Total_night_charge', 'Total_intl_minutes', 'Total_intl_calls', 'Total_intl_charge', 'Customer_service_calls']
correlation_matrix = combined_df[numerical_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


# ### Drop variables that have multicollinearity

# In[12]:


to_drop = ['Total_day_minutes', 'Total_eve_charge', 'Total_night_minutes', 'Total_intl_minutes']
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix.drop(columns=to_drop, errors='ignore').corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix after dropping variables that have multicollinearity')
plt.show()


#  ### Plotting class distribution (churned vs. not churned)

# In[13]:


plt.figure(figsize=(6, 6))
sns.countplot(x='Churn', data=combined_df)
plt.title('Class Distribution of Churn')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()


# In[14]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Churn', y='Total_day_minutes', data=combined_df)
plt.title('Total Day Minutes vs. Churn')
plt.xlabel('Churn')
plt.ylabel('Total Day Minutes')
plt.show()


# In[15]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Churn', y='Total_day_calls', data=combined_df)
plt.title('Total Day Calls vs. Churn')
plt.xlabel('Churn')
plt.ylabel('Total Day Calls')
plt.show()


# In[16]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Churn', y='Total_day_charge', data=combined_df)
plt.title('Total Day Charge vs. Churn')
plt.xlabel('Churn')
plt.ylabel('Total Day Charge')
plt.show()


# In[17]:


combined_df['International_plan'] = combined_df['International_plan'].astype(str)
combined_df['Churn'] = combined_df['Churn'].astype(str)
plt.figure(figsize=(10, 6))
sns.countplot(x=combined_df['International_plan'], hue=combined_df['Churn'])
plt.title('Churn by International Plan')
plt.xlabel('International Plan')
plt.ylabel('Count')
plt.show()


# In[18]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Voice_mail_plan', hue='Churn', data=combined_df)
plt.title('Churn by Voice Mail Plan')
plt.xlabel('Voice Mail Plan')
plt.ylabel('Count')
plt.show()


#  ### Plot a pie chart for the 'Churn' column

# In[19]:


combined_df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Churn Distribution')
plt.ylabel('') 
plt.show()


#  ### Plot bar chart for 'Customer service calls'

# In[20]:


combined_df['Customer_service_calls'].value_counts().sort_index().plot(kind='bar')
plt.title('Customer Service Calls')
plt.xlabel('Number of Calls')
plt.ylabel('Frequency')
plt.show()


# ### Plot the distribution of customer service calls

# In[21]:


sns.countplot(x='Customer_service_calls', data=combined_df)
plt.title('Distribution of Customer Service Calls')
plt.xlabel('Number of Calls')
plt.ylabel('Count')
plt.show()


# In[22]:


print('Description of the Categorical Dataset:')
combined_df.describe(include=['object', 'bool']).T


# # Phase Three : Preprocessing

# In[23]:


# Visualize the missing data
# Bar plot of missing values in each variable
msno.bar(combined_df)
plt.show()

# Matrix plot of missing data
msno.matrix(combined_df)
plt.show()

# Heatmap of missing data correlations
msno.heatmap(combined_df)
plt.show()

# Dendrogram of missing data
msno.dendrogram(combined_df)
plt.show()


# In[24]:


missing_values = combined_df.isnull().sum()
print(missing_values)


# ### Check for duplicates

# In[25]:


duplicates = combined_df.duplicated()
num_duplicates = duplicates.sum()
print("Number of duplicate rows:", num_duplicates)


# ## Convert column 'Churn' from bool to int

# In[26]:


# Replace 'False' with 0 and 'True' with 1
combined_df['Churn'] = combined_df['Churn'].replace({'False': 0, 'True': 1})

# Now convert the column to integers
combined_df['Churn'] = combined_df['Churn'].astype(int)


# ## Display a random sample of rows from the 'Churn'

# In[27]:


print(combined_df['Churn'].sample(15))


# In[28]:


def preprocess_data(df):
    df['International_plan'] = df['International_plan'].map({'Yes': 1, 'No': 0})
    df['Voice_mail_plan'] = df['Voice_mail_plan'].map({'Yes': 1, 'No': 0})
    return df


# In[29]:


combined_df = preprocess_data(combined_df)


# In[30]:


print(combined_df['International_plan'].unique())
print(combined_df['Voice_mail_plan'].unique())


# In[31]:


def feature_engineering(df):
    df['Total_minutes'] = df['Total_day_minutes'] + df['Total_eve_minutes'] + df['Total_night_minutes'] + df['Total_intl_minutes']
    df['Total_charges'] = df['Total_day_charge'] + df['Total_eve_charge'] + df['Total_night_charge'] + df['Total_intl_charge']
    return df


# In[32]:


combined_df = feature_engineering(combined_df)


# In[33]:


combined_df = pd.get_dummies(combined_df, columns=['State'])


# # Phase Four : Data splitting

# In[34]:


train_df = combined_df[combined_df['Data_Set'] == 'train'].drop(columns=['Data_Set'])
test_df = combined_df[combined_df['Data_Set'] == 'test'].drop(columns=['Data_Set'])


# In[35]:


X_train = train_df.drop(columns=['Churn'])
y_train = train_df['Churn']


# In[36]:


X_test = test_df.drop(columns=['Churn'])
y_test = test_df['Churn']


# # Phase Five : Training

# In[37]:


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# In[38]:


y_pred = model.predict(X_test)


# In[39]:


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[40]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[41]:


print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# In[42]:


y_test = y_test.astype(int)
y_pred = y_pred.astype(int)
print(f"Precision: {precision_score(y_test, y_pred)}")


# In[43]:


print(f"Recall: {recall_score(y_test, y_pred)}")


# In[44]:


y_test = y_test.astype(int)
y_pred = y_pred.astype(int)


# In[45]:


print(f"F1 Score: {f1_score(y_test, y_pred)}")


# In[46]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[47]:


knn_model = KNeighborsClassifier(n_neighbors=5)


# In[48]:


knn_model.fit(X_train, y_train)


# In[49]:


y_pred = knn_model.predict(X_test)


# In[50]:


y_pred_int = y_pred.astype(int)
print(classification_report(y_test, y_pred_int))


# In[51]:


from sklearn.preprocessing import StandardScaler


# In[52]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[53]:


logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)


# In[54]:


y_pred = logreg_model.predict(X_test_scaled)


# In[55]:


y_test = y_test.astype(int)
y_pred = y_pred.astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[56]:


report = classification_report(y_test, y_pred)


# In[57]:


print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(logreg_model)


# In[58]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)


# # Final Phase : Evaluation

# In[59]:


models = ['Decision Tree', 'KNN', 'Logistic Regression']
metrics = ['Precision (Class 0)', 'Recall (Class 0)', 'F1-score (Class 0)',
           'Precision (Class 1)', 'Recall (Class 1)', 'F1-score (Class 1)']
decision_tree_values = [0.98, 0.97, 0.98, 0.84, 0.88, 0.86]
knn_values = [0.90, 0.97, 0.93, 0.68, 0.34, 0.45]
logistic_regression_values = [0.89, 0.96, 0.92, 0.51, 0.25, 0.34]


# In[60]:


# Data for accuracy from the classification reports
models = ['Decision Tree', 'KNN', 'Logistic Regression']
accuracy_values = [0.96, 0.88, 0.86]  # Accuracy values for each model

# Plotting the bar chart for accuracy
plt.figure(figsize=(8, 5))
plt.bar(models, accuracy_values, color=['blue', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Classification Models')
plt.ylim(0.8, 1.0)  # Set the y-axis limits for better visualization
plt.show()


# In[61]:


# Data from the classification reports
models = ['Decision Tree', 'KNN', 'Logistic Regression']
metrics = ['Precision (Class 0)', 'Recall (Class 0)', 'F1-score (Class 0)',
           'Precision (Class 1)', 'Recall (Class 1)', 'F1-score (Class 1)']

# Values extracted from the reports
decision_tree_values = [0.98, 0.97, 0.98, 0.84, 0.88, 0.86]
knn_values = [0.90, 0.97, 0.93, 0.68, 0.34, 0.45]
logistic_regression_values = [0.89, 0.96, 0.92, 0.51, 0.25, 0.34]

# Plotting the bar chart with custom colors
fig, ax = plt.subplots(figsize=(10, 6))
index = range(len(metrics))

bar_width = 0.25
opacity = 0.8

# Custom colors for each model
colors = ['blue', 'green', 'red'] 

rects1 = plt.bar(index, decision_tree_values, bar_width,
                 alpha=opacity,
                 color=colors[0],
                 label='Decision Tree')

rects2 = plt.bar([p + bar_width for p in index], knn_values, bar_width,
                 alpha=opacity,
                 color=colors[1],
                 label='KNN')

rects3 = plt.bar([p + bar_width * 2 for p in index], logistic_regression_values, bar_width,
                 alpha=opacity,
                 color=colors[2],
                 label='Logistic Regression')

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Comparison of Classification Metrics by Model')
plt.xticks([p + bar_width for p in index], metrics, rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.show()

