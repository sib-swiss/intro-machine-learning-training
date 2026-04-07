%%time
X_penguin_train,X_penguin_test,y_penguin_train,y_penguin_test = train_test_split(X_penguin,y_penguin,stratify=y_penguin)

pipe = Pipeline([('scalar',StandardScaler()),("classifier", KNeighborsClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [
                {"classifier": [KNeighborsClassifier()],
                 "classifier__n_neighbors": np.arange(1,30,1),
                 "classifier__weights": ['uniform','distance']
                 },
                {"classifier": [LogisticRegression(class_weight='balanced', solver='saga')],
                 "classifier__l1_ratio": np.linspace(0,1,11),
                 "classifier__C": np.logspace(-2, 2, 10)
                 },
                {"classifier": [svm.SVC(class_weight='balanced', probability=True, kernel='linear')],
                 "classifier__C":np.logspace(-6, -1, 10)},
                {"classifier": [svm.SVC(class_weight='balanced', probability=True, kernel='rbf')],
                 "classifier__gamma": np.logspace(-2,1,10)},
                {"classifier": [svm.SVC(class_weight='balanced', probability=True, kernel='poly')],
                 "classifier__C":np.logspace(-6, -1, 10),
                 "classifier__degree":np.arange(2,10,1)}]

gridsearch_P = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1,scoring='roc_auc_ovr_weighted') # Fit grid search
best_model_P = gridsearch_P.fit(X_penguin_train,y_penguin_train)
print(best_model_P.best_params_)


## works for KNN as the best model

print("Model roc_auc_ovr_weighted:",best_model_P.score(X_penguin_test,y_penguin_test))

y_pred_test_c=best_model_P.predict(X_penguin_test)

bestw = best_model_P.best_params_['classifier__weights']
bestneighbors = best_model_P.best_params_['classifier__n_neighbors']


sns.heatmap( pd.crosstab( y_penguin_test, y_pred_test_c ) , 
             annot = True)
plt.xlabel("predicted")
plt.ylabel("observed")