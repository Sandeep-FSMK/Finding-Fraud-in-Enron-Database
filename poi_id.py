#!/usr/bin/python

import sys
import pickle
import pandas
import numpy 
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import mean


###  Automating the model selection / evaluation 

def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print "done.\n"
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    return mean(precision), mean(recall),features_train,labels_train






### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'deferral_payments', 'loan_advances', \
                 'bonus', 'restricted_stock_deferred', 'deferred_income', \
                 'expenses', 'exercised_stock_options', \
                 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',\
                 'to_messages', 'from_poi_to_this_person', \
                 'from_messages', 'from_this_person_to_poi', \
                  'shared_receipt_with_poi', 'total_payments', 'total_stock_value'] # You will need to use more features



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Look into the data 
print len(data_dict.keys())

##Convert the dictionary into a data frame 
df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))

# set the index of df to be the employees series:
df.set_index(employees, inplace=True)

#Pre-processing for 'NaN' values 

#Replacing the NaN values to 0 and then dropping the columns where the NaN values are greater than
#80 ( more than 50% of the dataset)
df.replace(to_replace='NaN', value=0, inplace=True)
df.drop(['loan_advances',
         'director_fees',
         'restricted_stock_deferred',
         'deferral_payments',
         'deferred_income',
         'long_term_incentive',
         'email_address'],axis=1,inplace=True)

df.apply(lambda x: pandas.to_numeric(x, errors='ignore'))
df=df.astype(float)
#print df.isnull().sum()
#print(df.shape) 
#print(df.head(20))
#print(df.describe())


#df.hist()
#plt.show()

# scatter plot matrix
#scatter_matrix(df)
#plt.show()
### Task 2: Remove outliers

## By auting the data in spreadsheet I could spot two ouliers - A total column and vendor - both cannot be a person of interest 
##so dropping both from the data_dict 

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']

for outlier in outliers:
    data_dict.pop(outlier,0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for person in my_dataset:
    try:
        
        
        total_messages = my_dataset[person]['from_messages'] + my_dataset[person]['to_messages']
        from_poi = my_dataset[person]["from_poi_to_this_person"]
        to_poi =  my_dataset[person]["from_this_person_to_poi"]
        shared_poi = my_dataset[person]["shared_receipt_with_poi"]
        poi_related_messages = from_poi +to_poi +shared_poi
                                                    
        #convert data types to float
        total_messages = float(total_messages)
        from_poi = float(from_poi)
        to_poi = float(to_poi)
        shared_poi = float(shared_poi)

        poi_ratio = poi_related_messages / total_messages
        my_dataset[person]['poi_ratio'] = poi_ratio

    except:
            
        my_dataset[person]['poi_ratio'] = 'NaN'

features_list = features_list + ['poi_ratio']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#print features_list

#Using the min-max scaler to the features 

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
#print features

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
k=8
k_best = SelectKBest(k=8)
k_best.fit(features, labels)
scores = k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
k_best_features = dict(sorted_pairs[:k])
print "{0} best features: {1}\n".format(k, k_best_features.keys())
#print k_best_features.keys()
print sorted_pairs

features_list = ['poi'] + k_best_features.keys()
print features_list

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.naive_bayes import GaussianNB
print "Gaussian NB Classifier"
gb_clf= GaussianNB()
clf = gb_clf
#evaluate_clf(clf,features,labels)
precision,recall,features_train,labels_train = evaluate_clf(clf,features,labels)

### Adaboost Classifier
print "Adaboost classifier"
from sklearn.ensemble import AdaBoostClassifier
a_clf = AdaBoostClassifier(algorithm= 'SAMME')
clf = a_clf
#evaluate_clf(clf,features,labels)

### Random Forest
print "Random Forest"
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
clf = rf_clf
#evaluate_clf(clf,features,labels)

### Decision Tree Classifier
print "Decision Tree Classifier"
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
clf = dt_clf 
#evaluate_clf(clf,features,labels)

print "Logistic Refression"
from sklearn.linear_model import LogisticRegression
#l_clf = LogisticRegression(C=10**18, tol=10**-21)
l_clf = LogisticRegression()
#clf=l_clf
#evaluate_clf(clf,features,labels)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)
    
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
clf = gb_clf
parameters_ada =  {'algorithm': ('SAMME', 'SAMME.R'),
 'base_estimator': [DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')],
 'learning_rate': [0.1, 0.5, 1, 1.5, 2, 2.5],
 'n_estimators': [5, 10, 30, 40, 50, 100, 150, 200],
 'random_state': [11]}

parameters_DT = { "min_samples_split":[2, 5, 10, 20],
                    "criterion": ('gini', 'entropy')
                    }
parameters_RF = {  "n_estimators":[2, 3, 5,200,700],
                            "criterion": ('gini', 'entropy'),
                 'max_features': ['auto', 'sqrt', 'log2']}
parameters_NB={
    
}
parameters_LR = {
         'penalty' : ['l1','l2'],

         'C' : [ 1e3,1e6,1e9,0.001, 0.01, 0.1, 1, 10, 100, 1000,10**18],
         'tol' : [ 1e-3,1e-6,1e-10,10**-21]
    }
#  Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5) 
#  Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters_NB,scoring=scorer)

#  Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(features_train, labels_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
#clf = best_clf
#evaluate_clf(clf,features,labels)

print "Gaussian Naive Bayes"
print (grid_fit.best_estimator_)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)