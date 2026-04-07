from sklearn.model_selection import train_test_split
## splitting the data into a train an test set
X_train, X_test , y_train , y_test = train_test_split( X , y , 
                                                      test_size=0.25, 
                                                      random_state= 668141 )


%%time
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from operator import itemgetter
# Linear model fitted by minimizing a regularized empirical loss with SGD (Stochastic gradient Descent)

pipeline_lr_reg_P=Pipeline([('poly',PolynomialFeatures()),
                            ('scalar',StandardScaler()),
                            ('model',SGDRegressor())])



# define the hyperparameters you want to test with their range
grid_values = {'poly__degree': [1,2],
               'poly__interaction_only': [True,False],
               'model__l1_ratio': np.linspace(0.0,1.0,11),
               'model__alpha':np.logspace(-1,2,10)}

# Feed them to GridSearchCV with the right score (R squared)
grid_lr_reg_P = GridSearchCV(pipeline_lr_reg_P, param_grid = grid_values, scoring='r2',n_jobs=-1)

grid_lr_reg_P.fit(X_train, y_train)

# get the best parameters
print('Grid best parameter (max. r2): ', grid_lr_reg_P.best_params_)
#get the best score calculated from the train/validation dataset
print('Grid best score (r2): ', grid_lr_reg_P.best_score_)

%%time
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.neighbors import KNeighborsRegressor
pipeline_knn_P=Pipeline([('selectk',SelectKBest(f_regression,k=500)),
                          ('scalar',StandardScaler()),
                          ('model', KNeighborsRegressor() )])

from sklearn.model_selection import GridSearchCV

# define the hyperparameters you want to test with their range
grid_values = {'selectk__k':np.arange(1,200,25),
               'model__n_neighbors': np.arange(3,10),
               'model__weights': ['uniform','distance'],
               }

# Feed them to GridSearchCV with the right score (R squared)
grid_knn_P = GridSearchCV(pipeline_knn_P, param_grid = grid_values, scoring='r2',n_jobs=-1)

grid_knn_P.fit(X_train, y_train)


# get the best parameters
print('Grid best parameter (max. r2): ', grid_knn_P.best_params_)
#get the best score calculated from the train/validation dataset
print('Grid best score (r2): ', grid_knn_P.best_score_)



%%time
from sklearn.ensemble import RandomForestRegressor

# define the hyperparameters you want to test with their range
grid_values = { 'model__n_estimators':[100,200,300], 
               'model__max_depth':[3,5,10,25],
               'model__min_samples_split':[5,10],
               'model__min_samples_leaf':[5,10]}


pipeline_RF_P=Pipeline([('select' , SelectKBest(f_regression, k=100)),
                        ('model',RandomForestRegressor())])

# Feed them to GridSearchCV with the right score (R squared)
grid_RF_P = GridSearchCV(pipeline_RF_P, param_grid = grid_values, scoring='r2',n_jobs=-1)

grid_RF_P.fit(X_train, y_train)


print('Grid best parameter (max. r2): ', grid_RF_P.best_params_)
print('Grid best score (r2): ', grid_RF_P.best_score_)


print('linear regression best score (r2):      ', grid_lr_reg_P.best_score_)
print('KNN best score (r2):                    ', grid_knn_P.best_score_)
print('Random Forest best score (r2):          ', grid_RF_P.best_score_)

# so here the grid which gives the best r2 is the LR
print( "best parameters", grid_lr_reg_P.best_params_ )

y_decision_fn_scores_acc=grid_lr_reg_P.score(X_test,y_test)

print('linear regression best parameter (max. r2) model on test: ', y_decision_fn_scores_acc)

m,M = min(y_train) , max(y_train)
plt.plot( [m,M] , [m,M] , color='black' , linestyle='dashed' )
plt.scatter( grid_lr_reg_P.best_estimator_.predict(X_train) ,  y_train, c='gray' , label='training points')
plt.scatter( grid_lr_reg_P.best_estimator_.predict(X_test) ,  y_test , c='xkcd:tomato' , label='testing points' )
plt.xlabel("predicted values")
plt.ylabel("true values")
plt.legend()

