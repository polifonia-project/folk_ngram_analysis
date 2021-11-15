# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-13T09:05:59.655163Z","iopub.execute_input":"2021-11-13T09:05:59.65564Z","iopub.status.idle":"2021-11-13T09:06:00.672013Z","shell.execute_reply.started":"2021-11-13T09:05:59.655513Z","shell.execute_reply":"2021-11-13T09:06:00.671Z"}}
#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

# reading the annotated dataset
df = pd.read_csv('../dataset/cre_root_detection.csv')


# finding out the missing data in the dataset
missing_data = pd.DataFrame({'total_missing': df.isnull().sum(), 'perc_missing': (df.isnull().sum()/82790)*100})
missing_data

# freq note and freq weighted acc data was found to missed at some places, and so replaced
# with frequent value of the same
df['freq note'].fillna(df['freq note'].mode()[0], inplace = True)
df['freq weighted acc'].fillna(df['freq weighted acc'].mode()[0], inplace = True)

# finding the null values
df.isnull().sum()

# just to get statistics of the data
df.describe()

# want to check the distribution of the following attributes and then draw the boxplot
num_cols = ['Krumhansl-Shmuckler','simple weights', 'Aarden Essen','Bellman Budge', 'Temperly Kostka Payne', 'as transcribed','final_note', 'freq note', 'freq weighted acc']
plt.figure(figsize=(18,9))
df[num_cols].boxplot()
plt.title("Numerical variables in Breathnach Ceol RinceBreathnach Ceol Rince Root detection dataset", fontsize=20)
plt.show()

# a function to visualize distribution of each variable
def plot_dist(col, ax):
    df[col][df[col].notnull()].value_counts().plot(kind='bar', facecolor='y', ax=ax)
    ax.set_xlabel('{}'.format(col), fontsize=20)
    ax.set_title("{} on Modcloth".format(col), fontsize= 18)
    return ax

f, ax = plt.subplots(3,3, figsize = (22,15))
f.tight_layout(h_pad=9, w_pad=2, rect=[0, 0.03, 1, 0.93])
cols = num_cols #['bra_size','bust', 'category', 'cup_size', 'fit', 'height', 'hips', 'length', 'quality']
k = 0
for i in range(3):
    for j in range(3):
        plot_dist(cols[k], ax[i][j])
        k += 1
__ = plt.suptitle("Initial Distributions of features", fontsize= 25)

# %% [markdown]
# **Explore 'expert assigned' variable**
df['expert assigned'].value_counts()

# draw expert assigned value
sns.countplot(x='expert assigned', data=df)

# %% [markdown]
# # select few columns for classification
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-13T09:06:03.321662Z","iopub.execute_input":"2021-11-13T09:06:03.321923Z","iopub.status.idle":"2021-11-13T09:06:03.327299Z","shell.execute_reply.started":"2021-11-13T09:06:03.321898Z","shell.execute_reply":"2021-11-13T09:06:03.326625Z"}}
df = df[["Krumhansl-Shmuckler", "simple weights","Aarden Essen", "Bellman Budge","Temperly Kostka Payne", "as transcribed", "final_note", 'freq note', 'freq weighted acc', 'expert assigned']]


# %% [markdown]
# # Declare feature vector and target variable
X = df.drop(['expert assigned'], axis=1)
y = df['expert assigned']

# drawing heatmap - correlation of attributes with target variable
plt.figure(figsize=(8,6))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.show()


# %% [markdown]
# # Split data into separate training and test set
# split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)

# check the shape of X_train and X_test
X_train.shape, X_test.shape

# %% [markdown]
# # Decision tree
from sklearn.tree import DecisionTreeClassifier

# instantiate the DecisionTreeClassifier model with criterion gini index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=56)

# fit the model
clf_gini.fit(X_train, y_train)


# %% [markdown]
#  # Predict the Test set results with criterion gini index

y_pred_gini = clf_gini.predict(X_test)

# %% [markdown]
# # Check accuracy score with criterion gini index

# importing relevant library
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

# %% [markdown]
# # Visualize decision-trees
plt.figure(figsize=(30,14))

from sklearn import tree

tree.plot_tree(clf_gini.fit(X_train, y_train))

# %% [markdown]
# # Decision Tree Classifier with criterion entropy
# instantiate the DecisionTreeClassifier model with criterion entropy
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)


# fit the model
clf_en.fit(X_train, y_train)

y_pred_en = clf_en.predict(X_test)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-13T09:06:20.042089Z","iopub.execute_input":"2021-11-13T09:06:20.042713Z","iopub.status.idle":"2021-11-13T09:06:20.05385Z","shell.execute_reply.started":"2021-11-13T09:06:20.042671Z","shell.execute_reply":"2021-11-13T09:06:20.052879Z"}}
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

# %% [markdown]
# # Random Forest
# import Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

# instantiate the classifier
rfc = RandomForestClassifier(random_state=0)


# fit the model
rfc.fit(X_train, y_train)

# Predict the Test set results
y_pred = rfc.predict(X_test)


# Check accuracy score
from sklearn.metrics import accuracy_score
print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# %% [markdown]
# # Random Forest Classifier model with parameter n_estimators=100


# instantiate the classifier with n_estimators = 100
rfc_100 = RandomForestClassifier(n_estimators=150, random_state=0)

# fit the model to the training set
rfc_100.fit(X_train, y_train)

# Predict on the test set results
y_pred_100 = rfc_100.predict(X_test)


# Check accuracy score
print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))

# %% [markdown]
# # Find important features with Random Forest model

# create the classifier with n_estimators = 100
clf = RandomForestClassifier(n_estimators=100, random_state=0)

# fit the model to the training set

clf.fit(X_train, y_train)

# %% [markdown]
# # Find important features with Random Forest model

# view the feature scores
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_scores

# %% [markdown]
# # Visualize feature scores of the features

# Creating a seaborn bar plot
sns.barplot(x=feature_scores, y=feature_scores.index)


# Add labels to the graph
plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

# Add title to the graph
plt.title("Visualizing Important Features")



# Visualize the graph
plt.show()
# %% [markdown]
# # Build Random Forest model on selected features
# declare feature vector and target variable
X = df.drop(['freq note', 'freq weighted acc'], axis=1)

y = df['expert assigned']
# %% [markdown]
# # split data into training and testing sets
# split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 30)

# %% [markdown]
# # #Randomforest model
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-13T09:06:20.980302Z","iopub.execute_input":"2021-11-13T09:06:20.981065Z","iopub.status.idle":"2021-11-13T09:06:21.019224Z","shell.execute_reply.started":"2021-11-13T09:06:20.981023Z","shell.execute_reply":"2021-11-13T09:06:21.018239Z"}}
# instantiate the classifier with n_estimators = 100
clf = RandomForestClassifier(n_estimators=10, random_state=0)

# fit the model to the training set
clf.fit(X_train, y_train)
# Predict on the test set results
y_pred = clf.predict(X_test)

# Check accuracy score
print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# %% [markdown]

# # Finally- Class wise evaluation - Confusion matrix

# Print the Confusion Matrix and slice it into four pieces
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))