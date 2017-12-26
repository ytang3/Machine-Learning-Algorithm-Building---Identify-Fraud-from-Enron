
# coding: utf-8

# # Project: Identify Fraud From Enron by Yiyi Tang

# In[47]:

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot
import pprint


# ### Background Information 

# In[48]:

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#To know background information about this data_dict
print "features of Skilling Jeffrey:", data_dict["SKILLING JEFFREY K"]
print "features number:", len(data_dict["SKILLING JEFFREY K"])
print "employee number:", len(data_dict)

n = 0
j = 0
for employee in data_dict:
    if data_dict[employee]["poi"]:
        n += 1        
print "number of poi:", n


# #### Find missing values in features

# In[49]:

from collections import Counter

count_missing_value = Counter()

for key, value in data_dict.items():
    for otherkey, othervalue in value.items():
        if othervalue == "NaN":
            count_missing_value[otherkey] += 1
        
print count_missing_value


# I'm surprised to see there's a lot missing values in the features, especially for "loan_advances", "director_fees" and restricted_stock_deferred". It is possible that most values of "director_fees" is "NaN". Because it is a feature describing cash payments for non-employee directors. Of course, most of employees are not director. In later section, I will use featureFormat function to replace these "NaN" values with 0.

# ### Task 1

# #### Remove outliers

# In[50]:

features = ["salary","bonus"]
remove_outliers_data = featureFormat(data_dict, features)

for point in remove_outliers_data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# In[51]:

###Remove outliers
data_dict.pop('TOTAL',0)


# In[52]:

### Seeking other outliers
for key in data_dict:
	if (data_dict[key]["bonus"] != 'NaN') and (data_dict[key]["salary"] != 'NaN'):
		if data_dict[key]["bonus"] > 5000000 and data_dict[key]["salary"] > 1000000:
			print key


# They're POI, and it makes sense their bonus and salary are extremely high. So I will not remove these important persons. 

# By manually looking at the original data (enron61702insiderpay.pdf), I noticed two person very suspect. One is THE TRAVEL AGENCY IN THE PARK. The other is LOCKHART,EUGENE E, whose features' values are all "NaN". Both of them is non POI, and is lack of values, so I decide to remove these two outliers.

# In[53]:

###Print suspectable person
print data_dict["LOCKHART EUGENE E"]


# In[54]:

###Print suspectable person
print data_dict["THE TRAVEL AGENCY IN THE PARK"]


# In[55]:

### Remove other outliers
data_dict.pop("LOCKHART EUGENE E",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)


# ### Task 2

# #### Create new feature

# In[56]:

### Create new feature called "fraction_to_poi" and add it to data_dict
for key in data_dict:
    if (data_dict[key]["from_this_person_to_poi"] not in ["NaN",0]) and (data_dict[key]["to_messages"] not in ["NaN",0]):
        data_dict[key]["fraction_to_poi"]= float(data_dict[key]["from_this_person_to_poi"]) / float(data_dict[key]["to_messages"])
    else:
        data_dict[key]["fraction_to_poi"]=0


# ### Task 3

# #### Feature selection

# In[57]:

features = ['poi','salary', 'deferral_payments', 'total_payments',                 'loan_advances', 'bonus', 'restricted_stock_deferred',                 'deferred_income', 'total_stock_value', 'expenses',                 'exercised_stock_options', 'other', 'long_term_incentive',                 'restricted_stock', 'director_fees', 'to_messages',                 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi', 'fraction_to_poi']

dt_data = featureFormat(data_dict, features, sort_keys = True)
dt_labels, dt_features = targetFeatureSplit(dt_data)

from sklearn.cross_validation import train_test_split
dt_features_train, dt_features_test, dt_labels_train, dt_labels_test = train_test_split(dt_features, dt_labels, test_size=0.3, random_state=42)


# In[58]:

print len(features)


# ##### Method 1: Feature importances with forests of tree

# In[59]:

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import numpy as np

# Build a classification task using 3 informative features
dt_features_train, dt_labels_train = make_classification(n_samples=146,
                           n_features=20)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier()

forest.fit(dt_features_train, dt_labels_train)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)

indices = np.argsort(importances)[::-1]

print("Feature ranking with Forests of Tree:")

for f in range(dt_features_train.shape[1]):
    print "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]), features[indices[f]+1]
    


# I noticed that everytime I run this method, it gives me totally different output of feature rankings. So I will go for another method instead of this.

# ##### Method 2: Decision tree based feature importances

# In[60]:

from sklearn import tree
import numpy
 
dt_clf = tree.DecisionTreeClassifier()
dt_clf = dt_clf.fit(dt_features_train, dt_labels_train)
 
#Identify the most powerful feature
importance = dt_clf.feature_importances_
 
#print importance
 
#print len(importance)
import numpy as np
indices = np.argsort(importance)[::-1]
print 'Feature Ranking with Decision Tree: '
for i in range(len(importance)):
    print "{} feature no.{}, {}, ({})".format(i+1, indices[i], features[indices[i]+1], importance[indices[i]]) 
 


# According to their rankings, **I find top 5 most powerful features: 'salary','bonus','total_stock_value', 'expenses', 'from_this_person_to_poi'.** (Notes: the feature rankings output is slightly different everytime re-run the codes. This might due to the selection of traning data)

# In[61]:

features_list = ['poi','total_stock_value','long_term_incentive','expenses','bonus']
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[62]:

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# ### Task 4

# #### Try different classifiers

# Using my testing script, I can see the performances of different algorithms I used below.

# #### Naive Bayes

# In[63]:

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

naive_clf = GaussianNB()
naive_clf.fit(features_train, labels_train)


# In[64]:

naive_pred = naive_clf.predict(features_test)
naive_accuracy = accuracy_score(naive_pred, labels_test)
print naive_accuracy


# Using testing script, the performance of Naive Bayes algorithm is as above. The precision score is 0.45052, the recall score is 0.32550, and F1 is 0.37794.

# #### Decision Tree

# In[65]:

from sklearn import tree


tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(features_train, labels_train)


# In[34]:

from sklearn.metrics import accuracy_score

tree_pred = tree_clf.predict(features_test)

tree_accuracy = accuracy_score(tree_pred, labels_test)

print tree_accuracy


# Using testing script, the performance of Decision tree algorithm is as above. The precision score is 0.36272, the recall score is 0.36000, and F1 is 0.36136.

# Thus, depending on the performances of above 3 algorithms, **I decide to use Naive Bayes. But I want to try tuning decision tree to see if its performance would be improved and be better than naive bayes.**

# ### Task 5

# Tuning decision tree

# In[79]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import numpy as np

param_grid = {'min_samples_split': [2,10],
             'max_depth': np.arange(3, 10),
             'criterion': ['gini','entropy']}
             

clf = GridSearchCV(DecisionTreeClassifier(), param_grid)

clf.fit(features_train, labels_train)
preds = clf.predict_proba(features_test)[:, 1]
performance = roc_auc_score(labels_test, preds)

print 'DecisionTree: Area under the ROC curve = {}'.format(performance)


# In[80]:

clf.best_params_


# In[81]:

clf = tree.DecisionTreeClassifier(min_samples_split=2,
                                 max_depth=5,
                                  criterion='gini')
                                 
clf.fit(features_train, labels_train)


# In[42]:

tree_pred = clf.predict(features_test)

tree_accuracy = accuracy_score(tree_pred, labels_test)

print tree_accuracy


# I tried to tune parameters like "min_samples_split", "max_depth", "class_weight", "splitter" and etc. The overall precision score are okay (around 0.38204), but the recall score are not as good as precision score, around 0.29150. I have to say: the performance of decision tree before tuning is better than its after tuning. But, the overall performance of naive bayes are the best among the algorithms I tried. 

# **Therefore, I decide to go for naive bayes for my final model.**

# In[83]:

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()
clf.fit(features_train, labels_train)


# In[84]:

pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)


# In[85]:



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

