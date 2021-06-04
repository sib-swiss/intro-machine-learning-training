
## first we bin prices, and get rid of the original price column

bins = np.quantile(df.price, [0,1/3.,2/3.,1])
group_names = ['Low', 'Medium', 'High']

df['price_binned'] = pd.cut(df['price'], bins, labels=group_names, include_lowest=True )
df.drop('price',axis = 1, inplace=True)

## separating target and coVariables 
X = df.drop( columns='price_binned' ) # notice the lack of in-place
y = df.price_binned

print(y.value_counts())

## transforming categorical variables 
X = pd.get_dummies(X)



## splitting in train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                  random_state=0)

grid_values = {'learning_rate':np.arange(0.01,0.1,0.01),
                'n_estimators':np.arange(1,500,100), 
               'max_depth':np.arange(2,int(len(X_train)),20),
               'min_samples_split':np.arange(2,int(len(X_train)/10),20),
              'min_samples_leaf':np.arange(1,int(len(X_train)/10),20)}# define the hyperparameters you want to test
#with the range over which you want it to be tested.

grid_tree_acc = GridSearchCV(GradientBoostingClassifier(), param_grid = grid_values, scoring='accuracy')#Feed it to the GridSearchCV with the right
#score over which the decision should be taken

grid_tree_acc.fit(X_train, np.array(y_train).ravel())

y_decision_fn_scores_acc=grid_tree_acc.score(X_test,np.array(y_test).ravel())

print('Grid best parameter (max. accuracy): ', grid_tree_acc.best_params_)#get the best parameters
print('Grid best score (accuracy): ', grid_tree_acc.best_score_)#get the best score calculated from the train/validation
#dataset
print('Grid best parameter (max. accuracy) model on test: ', y_decision_fn_scores_acc)# get the equivalent score on the test
#dataset : again this is the important metric

y_pred_test_c=grid_tree_acc.predict(X_test)

confusion_mc_c = confusion_matrix(y_test, y_pred_test_c)
df_cm_c = pd.DataFrame(confusion_mc_c, 
                     index = ['Low', 'Medium', 'High'], columns = ['Low', 'Medium', 'High'])

plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm_c, annot=True)
plt.title('Learning rate:'+str(grid_tree_acc.best_params_['learning_rate'])
          +' , n_estimators:'+str(grid_tree_acc.best_params_['n_estimators'])
          +' , max_depth:'+str(grid_tree_acc.best_params_['max_depth'])+' , min_split:'+str(grid_tree_acc.best_params_['min_samples_split'])+
          ' ,min_leaf:'+str(grid_tree_acc.best_params_['min_samples_leaf'])
          +'\nAccuracy:{0:.3f}'.format(accuracy_score(y_test, 
                                                                       y_pred_test_c)))
plt.ylim(3,0)
plt.ylabel('True label')
plt.xlabel('Predicted label')
from operator import itemgetter
RFC = GradientBoostingClassifier(learning_rate=grid_tree_acc.best_params_['learning_rate'],
                            n_estimators=grid_tree_acc.best_params_['n_estimators'],
                            max_depth=grid_tree_acc.best_params_['max_depth'],
                            min_samples_split=grid_tree_acc.best_params_['min_samples_split'],
                           min_samples_leaf=grid_tree_acc.best_params_['min_samples_leaf'])
RFC.fit(X_train, np.array(y_train).ravel())
w=RFC.feature_importances_#get the weights

sorted_features=sorted([[X.columns[i],abs(w[i])] for i in range(len(w))],key=itemgetter(1),reverse=True)

print('Features sorted per importance in discriminative process (up to 99% of cumulated weight)')
s = 0
for f,w in sorted_features:
    print(f.rjust(25), '{:.4f}'.format(w) , sep='\t')
    s += w
    if s> 0.99:
        break
