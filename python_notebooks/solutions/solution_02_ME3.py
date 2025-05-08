pipeline_lr=Pipeline([('scalar',StandardScaler()), 
                      ('model',LogisticRegression(class_weight='balanced',solver='liblinear'))])



grid_values = {'model__C': np.logspace(-5,2,200),
               'model__penalty': ['l1','l2'] }
# define the hyperparameters you want to test
# with the range over which you want it to be tested.


# Feed it to the GridSearchCV with the right
# score(here accuracy) over which the decision should be taken
grid_lr_acc = GridSearchCV(pipeline_lr, 
                           param_grid = grid_values, 
                           scoring='balanced_accuracy',
                           cv=10, 
                           n_jobs=-1)


## this cell throws a lot of warning, I remove them with the lines under
grid_lr_acc.fit(X_cancer_train, y_cancer_train)
print('Grid best parameter (max. balanced accuracy): ', grid_lr_acc.best_params_)#get the best parameters
print('Grid best score (cross-validated balanced accuracy): {:.4f}'.format( grid_lr_acc.best_score_ ) )
