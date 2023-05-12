%%time

pipe = Pipeline([('scalar',StandardScaler()),("classifier", KNeighborsClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [
                {"classifier": [KNeighborsClassifier(n_jobs=-1)],
                 "classifier__n_neighbors": np.arange(1,30,5),
                 "classifier__weights": ['uniform','distance']
                 },
                {"classifier": [LogisticRegression(n_jobs=1,class_weight='balanced', solver='liblinear')],
                 "classifier__penalty": ['l2','l1'],
                 "classifier__C": np.logspace(-2, 2, 10)
                 },
                {"classifier": [svm.SVC(class_weight='balanced', probability=True, kernel='linear')],
                 "classifier__C":np.logspace(-6, -1, 10)},
                {"classifier": [svm.SVC(class_weight='balanced', probability=True, kernel='rbf')],
                 "classifier__gamma": np.logspace(-2,1,10)},
                {"classifier": [svm.SVC(class_weight='balanced', probability=True, kernel='poly')],
                 "classifier__C":np.logspace(-6, -1, 10),
                 "classifier__degree":np.arange(2,10,1)}]

gridsearch_C = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1,scoring='roc_auc') # Fit grid search
best_model_C = gridsearch_C.fit(X_cancer_train,y_cancer_train)

print(best_model_C.best_params_)
print("Model accuracy:",gridsearch_C.best_score_)


## predicting the labels on the test set    
y_pred_test_c=best_model_C.predict(X_cancer_test)

bestC = best_model_C.best_params_['classifier__gamma']



plotTitle = 'RBF: gamma: {:.1e}\n Accuracy: {:.3f}'.format(bestGamma,
                                                         accuracy_score(y_cancer_test,y_pred_test_c) )


plotConfusionMatrix( y_cancer_test, y_pred_test_c, 
                    ['Benign','Malignant'] , plotTitle , 
                    ax = None)


from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(best_model_C,X_cancer_test, y_cancer_test)


## solution with PCA
%%time
from sklearn.decomposition import PCA

grid_param = [
                {"classifier": [KNeighborsClassifier(n_jobs=-1)],
                 "classifier__n_neighbors": np.arange(1,30,5),
                 "classifier__weights": ['uniform','distance']
                 },
                {"classifier": [LogisticRegression(n_jobs=1,class_weight='balanced', solver='liblinear')],
                 "classifier__penalty": ['l2','l1'],
                 "classifier__C": np.logspace(-2, 2, 10)
                 },
                {"classifier": [svm.SVC(class_weight='balanced', probability=True, kernel='linear')],
                 "classifier__C":np.logspace(-6, -1, 10)},
                {"classifier": [svm.SVC(class_weight='balanced', probability=True, kernel='rbf')],
                 "classifier__gamma": np.logspace(-2,1,10)},
                {"classifier": [svm.SVC(class_weight='balanced', probability=True, kernel='poly')],
                 "classifier__C":np.logspace(-6, -1, 10),
                 "classifier__degree":np.arange(2,10,1)}]



PCA_NCOMPONENTS = 5

pipe_pca = Pipeline([('scalar1',StandardScaler()),
                     ('pca',PCA(n_components=PCA_NCOMPONENTS)),
                     ("classifier", svm.SVC(class_weight='balanced'))])
# Create dictionary with candidate learning algorithms and their hyperparameters

# create a gridsearch of the pipeline, the fit the best model
gridsearch_c_pca = GridSearchCV(pipe_pca, 
                                grid_param, scoring='roc_auc',
                                cv=5, verbose=0,n_jobs=-1) # Fit grid search

best_model_c_pca = gridsearch_c_pca.fit(X_cancer_train,y_cancer_train)

print(best_model_c_pca.best_params_)
print("Model accuracy:",gridsearch_c_pca.best_score_)


## predicting the labels on the test set    
y_pred_test_c=best_model_c_pca.predict(X_cancer_test)

bestC = best_model_C.best_params_['classifier__C']
bestPenalty = best_model_C.best_params_['classifier__penalty']

plotTitle = 'logistic regression: {} penalty ; C: {:.1e}\n Accuracy: {:.3f}'.format(bestPenalty,
                                                                         bestC,
                                                                         accuracy_score(y_cancer_test,y_pred_test_c) )


plotConfusionMatrix( y_cancer_test, y_pred_test_c, 
                    ['Benign','Malignant'] , plotTitle , 
                    ax = None)


from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(best_model_C,X_cancer_test, y_cancer_test)
