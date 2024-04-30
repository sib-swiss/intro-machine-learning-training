grid_values = {'criterion': ['entropy','gini'],
               'max_depth':[2,6,10,14],
               'min_samples_split':np.arange(2,len(X_penguin_train)//4,5),
              'min_samples_leaf':np.arange(1,len(X_penguin_train)//4,5)}


grid_tree_p_roc_auc = GridSearchCV(DecisionTreeClassifier(), 
                             param_grid = grid_values, 
                             scoring='roc_auc_ovr_weighted',
                             n_jobs=-1)

grid_tree_p_roc_auc.fit(X_penguin_train, y_penguin_train)

y_decision_fn_scores_p_roc_auc=grid_tree_p_roc_auc.score(X_penguin_test,y_penguin_test)

print(f'Grid best score (roc_auc_ovr_weighted): {grid_tree_p_roc_auc.best_score_:.3f}')
print('Grid best parameter (max. roc_auc_ovr_weighted): ')
for k,v in grid_tree_p_roc_auc.best_params_.items():
    print('\t',k,'->',v)
    
from sklearn.metrics import accuracy_score, confusion_matrix

y_test_score=grid_tree_p_roc_auc.score(X_penguin_test,y_penguin_test)

print('Grid best parameter (max. accuracy) model on test: ', y_test_score)

y_penguin_pred_test = grid_tree_p_roc_auc.predict(X_penguin_test)

confusion_m_cancer = confusion_matrix(y_penguin_test, y_penguin_pred_test)

plt.figure(figsize=(5.5,4))
sns.heatmap(confusion_m_cancer, annot=True)
plt.title('test {} : {:.3f}'.format( grid_tree_p_roc_auc.scoring , y_test_score ))
plt.ylabel('True label')
plt.xlabel('Predicted label')