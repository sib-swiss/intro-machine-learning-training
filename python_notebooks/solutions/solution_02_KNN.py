from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X_penguin_train, X_penguin_test, y_penguin_train, y_penguin_test = train_test_split(
                                                    X_penguin, y_penguin,
                                                    random_state=4212280,stratify=y_penguin)

knn_i=KNeighborsClassifier(n_jobs=-1)

pipeline_knn_i=Pipeline([('scalar',StandardScaler()),
                     ('model',knn_i)])


# define the hyperparameters you want to test
#with the range over which you want it to be tested.
grid_values = {'model__n_neighbors': np.arange(5,50,1),
               'model__weights':['uniform','distance']}

#Feed it to the GridSearchCV with the right
#score over which the decision should be taken
grid_knn_acc_Penguin = GridSearchCV(pipeline_knn_i, 
                                param_grid = grid_values, 
                                scoring='accuracy',n_jobs=-1)

grid_knn_acc_Penguin.fit(X_penguin_train, y_penguin_train)


testBestScore =grid_knn_acc_Penguin.score(X_penguin_test,y_penguin_test)

print('Grid best parameter (max. accuracy):\n\t', grid_knn_acc_Penguin.best_params_)
print('Grid best score (accuracy): {:.3f}'.format( grid_knn_acc_Penguin.best_score_) )

print('Grid best parameter (max. accuracy) model on test: {:.3f}'.format( testBestScore) )

## predicting the labels on the test set    
y_pred_test=grid_knn_acc_Penguin.predict(X_penguin_test)

bestNN = grid_knn_acc_Penguin.best_params_['model__n_neighbors']
bestWeight = grid_knn_acc_Penguin.best_params_['model__weights']

plotTitle = 'KNN n_neighbors: {}, weights: {}\n Accuracy: {:.3f}'.format(bestNN,
                                                                         bestWeight,
                                                                         accuracy_score(y_penguin_test,y_pred_test) )

confusion_m_penguin = confusion_matrix(y_penguin_test, y_pred_test)
plt.figure(figsize=(5.5,4))
sns.heatmap(confusion_m_penguin, annot=True)
plt.title(plotTitle)
plt.ylabel('True label')
plt.xlabel('Predicted label')
