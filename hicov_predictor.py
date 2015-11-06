import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation

def split_data(data, predict_cols, n_samples=False):
    predict_cols.append('hicov') 
    
    if not n_samples: 
        train_data = data[data.hicov.notnull()][predict_cols].dropna() 
    else: 
        train_data = data[data.hicov.notnull()][predict_cols].sample(n_samples).dropna() 
        
    predict_cols.pop()  # Remove 'hicov'

    X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
    train_data[predict_cols].values, \
    train_data.hicov.values, test_size=0.4)

    return X_train, X_test, y_train, y_test 
    

#%% Read in data 
print "Reading in the .txt file..."

data = pd.read_csv('coverage_data.txt.gz', header=0, sep="\t", index_col=0)

print "Size of data frame: ", data.shape
print "%.1f million rows" % (data.shape[0]/1.0e6)


#%% Linear Model
from sklearn import linear_model

#Most predictive features from random forest
predict_cols = ['agep', 'bld', 'cit', 'dis',  'fs', 'hht', 'hincp', 'mar',\
      'noc', 'np', 'puma','rac1p', 'sch', 'sex', 'st']
X_train, X_test, y_train, y_test = split_data(data, predict_cols)


clf = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', verbose=1)
clf.fit(X_train, y_train)

print "Logistic Regression: ", clf.score(X_test, y_test)

#%% Decision Tree
from sklearn import tree
#from sklearn.externals.six import StringIO  
#import pydot_ng as pydot

predict_cols = ['agep', 'bld', 'cit', 'dis',  'fs', 'hht', 'hincp', 'mar',\
      'noc', 'np', 'puma','rac1p', 'sch', 'sex', 'st']

#predict_cols = ['agep', 'bld', 'cit', 'dis',  'fs', 'hht', 'hincp', 'mar',\
#     'mv', 'noc', 'np', 'puma','rac1p', 'sch', 'sex', 'st', 'type', 'veh']

X_train, X_test, y_train, y_test = split_data(data, predict_cols)

depths = range(1,15)
scores = []
for depth in depths: 
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
    

plt.plot(depths, scores)
plt.xlabel('Tree Depth')
plt.ylabel('Score')
#dot_data = StringIO() 
#tree.export_graphviz(clf, out_file=dot_data) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 


#%% Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

predict_cols = ['agep', 'bld', 'cit', 'dis',  'fs', 'hht', 'hincp', 'mar',\
     'mv', 'noc', 'np', 'puma','rac1p', 'sch', 'sex', 'st', 'type', 'veh']

X_train, X_test, y_train, y_test = split_data(data, predict_cols)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,\
     max_depth=3, max_features=11)
     
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

#%% Print results from Linear Model
from sklearn import linear_model
clf = linear_model.LogisticRegression()

predict_cols = ['agep', 'bld', 'cit', 'dis',  'fs', 'hht', 'hincp', 'mar',\
     'mv', 'noc', 'np', 'puma','rac1p', 'sch', 'sex', 'st', 'type', 'veh']

predict_cols.append('hicov') 
train_data = data[data.hicov.notnull()][predict_cols].dropna()
predict_cols.pop()  # Remove 'hicov'


X_train, y_train = train_data[predict_cols], train_data.hicov
clf.fit(X_train, y_train)

# Predict scores for hicov == NaN 
predict_data = data[data.hicov.isnull()][predict_cols].dropna()

predicted_proba = clf.predict_proba(predict_data)

predict_data['probability_score'] = predicted_proba[:,0]

predict_data['probability_score'].to_csv('roryhartongredden_datascience1_scores.csv', \
    header=True)