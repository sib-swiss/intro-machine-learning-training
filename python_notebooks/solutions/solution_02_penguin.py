%%time
pipe = Pipeline([('scalar',StandardScaler()),("classifier", KNeighborsClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [
                {"classifier": [KNeighborsClassifier(n_jobs=-1)],
                 "classifier__n_neighbors": np.arange(1,30,1),
                 "classifier__weights": ['uniform','distance']
                 },
                {"classifier": [LogisticRegression(n_jobs=-1,class_weight='balanced')],
                 "classifier__penalty": ['l2','l1'],
                 "classifier__C": np.logspace(-2, 2, 10)
                 },
                {"classifier": [svm.SVC(class_weight='balanced')],
                 "classifier__kernel": ['linear', 'rbf', 'poly'],
                 "classifier__C":np.logspace(-6, -1, 10),
                 "classifier__degree":np.arange(0,10,1),
                 "classifier__gamma": np.logspace(-2,1,10)}]

gridsearch_P = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1,scoring='roc_auc_ovr_weighted') # Fit grid search
best_model_P = gridsearch_P.fit(X_penguin_train,y_penguin_train)


## works for KNN as the best model
print(best_model_P.best_params_)
print("Model roc_auc_ovr_weighted:",best_model_P.score(X_penguin_test,y_penguin_test))

y_pred_test_c=best_model_P.predict(X_penguin_test)

bestw = best_model_P.best_params_['classifier__weights']
bestneighbors = best_model_P.best_params_['classifier__n_neighbors']


plotTitle = 'KNN weights: {}, n_neighbors: {},\n Accuracy: {:.5f}'.format(bestw,
                                                                         bestneighbors,
                                                                         accuracy_score(y_penguin_test,y_pred_test_c) )


plotConfusionMatrix( y_penguin_test, y_pred_test_c, 
                    ['Adelie','Chinstrap','Gentoo'] , plotTitle , 
                    ax = None)

