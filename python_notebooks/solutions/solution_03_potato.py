## splitting the data into a train an test set
X_train, X_test , y_train , y_test = train_test_split( dfTT , y , 
                                                      test_size=0.25, 
                                                      stratify = y , random_state= 668141 )


## Counter is a nice class which takes a list and creates a dictionnary 
# whose keys are the unique list items and values are the number of time they appear
from collections import Counter
print( "train set", Counter( y_train ) )
print( "test set", Counter( y_test ) )


%%time
##the %%time is a jupyter cell magic command which will measure the time 
# it takes for a cell to run and report it.
# WARNING : it only works if it is the 1st line of the cell, so you have to manually move it there...

# starting with a feature selection
skb = SelectKBest(chi2, k=20)
skb.fit(X_train , y_train)
X_train_reduced = X_train.loc[ : , skb.get_support() ]

X_test_reduced = X_test.loc[ : , skb.get_support() ]

# training pipeline
pipe = Pipeline([("classifier", DecisionTreeClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [ {'classifier':[RandomForestClassifier(n_jobs=-1,class_weight='balanced')],
                'classifier__criterion': ['entropy','gini'],
                'classifier__n_estimators':np.arange(1,1000,100), 
                'classifier__max_depth':[2],
                'classifier__min_samples_split':[2],
                'classifier__min_samples_leaf':[1]},
              {"classifier": [GradientBoostingClassifier()],
                'classifier__learning_rate':np.arange(0.01,0.1,0.02),
                'classifier__n_estimators':np.arange(1,100,20), 
                'classifier__max_depth':[2],
                'classifier__min_samples_split':[2],
                'classifier__min_samples_leaf':[1]},
              {'classifier':[AdaBoostClassifier()],
               'classifier__n_estimators':np.arange(1,1000,200), 
               'classifier__learning_rate':np.arange(0.01,0.1,0.02) }]



gridsearch_Potato = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1,scoring='roc_auc') # Fit grid search
best_model_Potato = gridsearch_Potato.fit(X_train_reduced,y_train)

print(best_model_Potato.best_params_)
print("Model roc_auc on test set:",best_model_Potato.score(X_test_reduced,y_test))


## predicting the labels on the test set    
y_pred_test=best_model_Potato.predict(X_test_reduced)

bestN = best_model_Potato.best_params_["classifier__n_estimators"]
bestCrit = best_model_Potato.best_params_["classifier__criterion"]
bestMD = best_model_Potato.best_params_["classifier__max_depth"]
bestMSL = best_model_Potato.best_params_["classifier__min_samples_leaf"]
bestMSS = best_model_Potato.best_params_["classifier__min_samples_split"]


plotTitle = """Random Forest - number of estimators :{}
criterion: {}, max depth: {}
min_samples_leaf: {}, min_samples_split: {},
Accuracy: {:.3f}""".format(bestN,bestCrit,bestMD,bestMSL,bestMSS,
                           accuracy_score(y_test,y_pred_test) )


plotConfusionMatrix( y_test, y_pred_test, 
                    ['White','Yellow'],
                    plotTitle , 
                    ax = None)
plot_roc_curve(best_model_Potato,X_test_reduced, y_test)

## extract the best estimator steps from the pipeline
RF = best_model_Potato.best_estimator_.steps[0][1]

w=RF.feature_importances_#get the weights
selectedFeatures = dfTT.columns[ skb.get_support() ]

featureW = pd.DataFrame( {'feature': selectedFeatures,'weight':w} )

# sort them by absolute value
featureWsorted = featureW.sort_values(by=['weight'] , 
                                      ascending=False , 
                                      key=lambda col : col.abs())

# get the non-null ones
print('Features sorted per importance:')
featureWsorted.loc[ ~ np.isclose( featureWsorted["weight"] , 0 ) ]
