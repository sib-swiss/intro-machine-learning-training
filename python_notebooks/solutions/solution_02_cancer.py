%%time
pipe = Pipeline([('scalar',StandardScaler()),("classifier", KNeighborsClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [
                {"classifier": [KNeighborsClassifier(n_jobs=-1)],
                 "classifier__n_neighbors": np.arange(1,30,5),
                 "classifier__weights": ['uniform','distance']
                 },
                {"classifier": [LogisticRegression(n_jobs=-1,class_weight='balanced')],
                 "classifier__penalty": ['l2','l1'],
                 "classifier__C": np.logspace(-2, 2, 10)
                 },
                {"classifier": [svm.SVC(class_weight='balanced')],
                 "classifier__kernel": ['linear', 'rbf', 'poly'],
                 "classifier__C":np.logspace(-6.5, -1.5, 10),
                 "classifier__degree":np.arange(0,10,1),
                 "classifier__gamma": np.logspace(-2,1,10)}]

gridsearch_C = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1,scoring='roc_auc') # Fit grid search
best_model_C = gridsearch_C.fit(X_cancer_train,y_cancer_train)

print(best_model_C.best_params_)
print("Model roc_auc:",best_model_C.score(X_cancer_test,y_cancer_test))

##the best model here is a SVC

## predicting the labels on the test set    
y_pred_test_c=best_model_C.predict(X_cancer_test)

bestC = best_model_C.best_params_['classifier__C']
bestKernel = best_model_C.best_params_['classifier__kernel']
bestDeg = best_model_C.best_params_['classifier__degree']

plotTitle = 'SVC kernel: {} - degree: {}, C: {}\n Accuracy: {:.3f}'.format(bestKernel,
                                                                           bestDeg,
                                                                         bestC,
                                                                         accuracy_score(y_cancer_test,y_pred_test_c) )


plotConfusionMatrix( y_cancer_test, y_pred_test_c, 
                    ['Benign','Malignant'] , plotTitle , 
                    ax = None)


plot_roc_curve(best_model_C,X_cancer_test, y_cancer_test)


%%time

from sklearn.decomposition import PCA

grid_param = {"classifier__kernel": ['poly'],
                 "classifier__C":np.logspace(-2, 2, 10),
                 "classifier__degree":np.arange(0,10,1)}



PCA_NCOMPONENTS = 4

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
print("Model accuracy:",best_model_c_pca.score(X_cancer_test,y_cancer_test))


## predicting the labels on the test set    
y_pred_test_c=best_model_c_pca.predict(X_cancer_test)

bestC = best_model_c_pca.best_params_['classifier__C']
bestKernel = best_model_c_pca.best_params_['classifier__kernel']
bestDeg = best_model_c_pca.best_params_['classifier__degree']

plotTitle = 'SVC kernel: {} - degree: {}, C: {}\n Accuracy: {:.3f}'.format(bestKernel,
                                                                           bestDeg,
                                                                         bestC,
                                                                         accuracy_score(y_cancer_test,y_pred_test_c) )


plotConfusionMatrix( y_cancer_test, y_pred_test_c, 
                    ['Benign','Malignant'] , plotTitle , 
                    ax = None)

plot_roc_curve(best_model_c_pca,X_cancer_test, y_cancer_test)



