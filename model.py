#!/usr/bin/env python
# coding: utf-8


# # Importing the data

from copyreg import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the scraped results from above
novels_raw = pd.read_csv('results_novels.csv')


# # Data cleaning
# making the ratings and review_count column numeric
novels_raw['rating'] = novels_raw['rating'].str.replace('out of 5 stars', '').astype(float)
novels_raw['review_count'] = novels_raw['review_count'].astype(str).astype(int)
novels_raw.tail(60)       


# adding a new cleaner author column, 'author2' while removing unwanted data like 'Book 6 of 6:' from the author column
novels_raw['author2'] = novels_raw['author'].str.split(':', n=1).str.get(-1).str.strip()
novels_raw.head(5)

#checking for duplicates
novels_raw.describe(include='object')

# Although there are 510 titles, there are only 500 unique titles, which means some of them are duplicates, which have to be dropped.
novels_raw = novels_raw.drop_duplicates(subset=['description'])

# Renaming and reorganizing the columns
novels_raw.rename(columns = {'description':'title', 'review_count':'customers_rated', 'bestseller': 'bestseller/editorspick'}, inplace=True)
novels_clean = novels_raw[['title', 'author2', 'binding', 'rating', 'customers_rated',  'price', 'bestseller/editorspick', 'url']]
novels_clean

# Replacing the 'bestseller/editorspick' column with TRUE or FALSE
novels_clean['bestseller/editorspick'] = novels_clean['bestseller/editorspick'].replace(["Editors' pick", 'Best Seller', 'Goodreads Choice', "Teachers' pick"], 'bestseller')
novels_clean['bestseller/editorspick'] = novels_clean['bestseller/editorspick'].replace(['No'], 'non-bestseller')
novels_clean                                                                                         


# Getting data for only the bestselling novels
bestsellers = novels_clean.loc[novels_clean['bestseller/editorspick'] == 'TRUE']

# Getting the authors with atleast one best seller novel
bestsellers_authors = bestsellers['author2'].drop_duplicates()

# Getting the average bestseller number of reviews by book format
bestsellers.pivot_table('customers_rated', index='binding', aggfunc='mean')

# Getting the data for the novel with the highest price
novel_maxprice = novels_clean.loc[novels_clean['price'] == novels_clean['price'].max()]

# Getting the data for the novel with the lowest price excluding $0 items since most Kindle and Audiobooks are listed at $0 and are based on subscription
novel_minprice = novels_clean.loc[novels_clean['price'] == novels_clean['price'][novels_clean['price'].gt(0)].min(0)]


# ## Data preparation for modeling
# Here, we want to build a machine learning classifier to predict whether a novel will be a bestseller on amazon.com based on the attibutes, author, binding, rating, customers_rated (reviews) and price.

# Selecting the data to be used
novels = novels_clean[['binding', 'rating', 'customers_rated', 'price', 'bestseller/editorspick']]

#One-hot-encoding the categorical values

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

novels_onehot1 = pd.get_dummies(novels['binding'], prefix='format', drop_first=True)
novels_onehot = novels.join(novels_onehot1).drop(['binding'], axis=1)

column_names = ['rating', 'customers_rated', 'price', 'format_Hardcover', 'format_Kindle', 'format_Mass Market Paperback', 'format_Paperback', 'bestseller/editorspick']
novels_onehot = novels_onehot.reindex(columns = column_names)
novels_onehot

# Splitting the data into features, X and the response, y

X = novels_onehot.iloc[:, :-1]
y = novels_onehot.iloc[:, -1]

# Splitting the data into the train sets and the test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


# ## Selecting and training a model

# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []

# #### Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(X_train, y_train)

predicted_values = DecisionTree.predict(X_test)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)


# #### Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=20, random_state=0)
rfc.fit(X_train,y_train)

predicted_values = rfc.predict(X_test)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('rfc')
print("rfc's Accuracy is: ", x)


# #### XGBoost Classifier

import xgboost as xgb
xb = xgb.XGBClassifier()

y_train_encoded = label_encoder.fit_transform(y_train) #encoding the y values since model can only accept 0 & 1
y_test_encoded = label_encoder.fit_transform(y_test)
y_encoded = label_encoder.fit_transform(y)

xb.fit(X_train, y_train_encoded)

predicted_values = xb.predict(X_test)

x = metrics.accuracy_score(y_test_encoded, predicted_values)
acc.append(x)
model.append('xb')
print("XGBoost's Accuracy is: ", x)


# #### Supprt Vector Machine
# The customers_rated column has a relativley wide range of values from 0 to thousands. Variables with bigger magnitude will dominate over those with smaller magnitude. Thus feature scaling is needed here to bring the column to relative values comparable to the other numeric columns, since some ML algorithms are sensitive to the relative scales of features.

from sklearn.svm import SVC
# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# fit transform scaler on training data
norm = MinMaxScaler()#.fit(X_train)
X_train_norm = norm.fit_transform(X_train)

# transform testing data
X_test_norm = norm.transform(X_test)

svm = SVC(kernel='poly', degree=3, C=1)
svm.fit(X_train_norm, y_train)

predicted_values = svm.predict(X_test_norm)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('svm')
print("svm's Accuracy is: ", x)

# If svm is chosen, there will be no need to reverse normalization on the predictions since the y data was not scaled


# ### Model accuracy comparison

plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc, y = model, palette='tab20')

# From the accuracy figures we see that the best model is the Random forest Classifier with an accuracy of 75%.

# ### Predicting the bestseller novels from the test data

y_pred = rfc.predict(X_test)
y_pred


# # Model Deployment

# ### Export model to pickle file

import pickle

#saving the trained model to a pickle file
pickle.dump(rfc, open('finalModel.pkl', 'wb'))

#loading the model to test it
model = pickle.load(open("finalModel.pkl", "rb"))






