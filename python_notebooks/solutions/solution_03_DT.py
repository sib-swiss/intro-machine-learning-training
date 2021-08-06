grid_values = {'criterion': ['entropy','gini'],
               'max_depth':np.arange(2,len(X_penguin_train),10),
               'min_samples_split':np.arange(2,len(X_penguin_train)//4,5),
              'min_samples_leaf':np.arange(1,len(X_penguin_train)//10,5)}


grid_tree_p_roc_auc = GridSearchCV(DecisionTreeClassifier(), 
                             param_grid = grid_values, 
                             scoring='roc_auc_ovr_weighted',
                             n_jobs=-1)

grid_tree_p_roc_auc.fit(X_penguin_train, y_penguin_train)

y_decision_fn_scores_p_roc_auc=grid_tree_p_roc_auc.score(X_penguin_test,y_penguin_test)

print('Grid best parameter (max. roc_auc_ovr_weighted): ', grid_tree_p_roc_auc.best_params_)
print('Grid best score (roc_auc_ovr_weighted): ', grid_tree_p_roc_auc.best_score_)
print('Grid best parameter (max. roc_auc_ovr_weighted) model on test: ', y_decision_fn_scores_p_roc_auc)
y_pred_test=grid_tree_p_roc_auc.predict(X_penguin_test)

bestCrit = grid_tree_p_roc_auc.best_params_["criterion"]
bestMD = grid_tree_p_roc_auc.best_params_["max_depth"]
bestMSL = grid_tree_p_roc_auc.best_params_["min_samples_leaf"]
bestMSS = grid_tree_p_roc_auc.best_params_["min_samples_split"]


plotTitle = """decision tree criterion: {}, max depth: {}
min_samples_leaf: {}, min_samples_split: {},
Accuracy: {:.3f}""".format(bestCrit,bestMD,bestMSL,bestMSS,
                           accuracy_score(y_penguin_test,y_pred_test) )


plotConfusionMatrix( y_penguin_test, y_pred_test, 
                    ['Adelie','Chinstrap','Gentoo'] , plotTitle , 
                    ax = None)


