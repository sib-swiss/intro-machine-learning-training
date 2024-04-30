## importing stuff
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split , GridSearchCV, StratifiedKFold 
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

## looking at the data a bit 
print("TenYearCHD value counts:")
print( df_heart.TenYearCHD.value_counts() )
print("we can see there is a lot of imbalance")
print("\n***\n")
print("Fraction of NAs:")
print( df_heart.isna().mean() )
print("some NA, but not a huge amount")
print("\n***\n")

## splitting train and test
X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(X_heart,y_heart,
                                                                            stratify=y_heart)
#stratify is here to make sure that you split keeping the repartition of labels unaffected

print(f"full : {sum(y_heart)} CHD / {len(y_heart)} samples")
print(f"train: {sum(y_heart_train)} CHD / {len(y_heart_train)} samples")
print(f"test : {sum(y_heart_test)} CHD / {len(y_heart_test)} samples")
print("\n***\n")

## creating the pipeline and grid, and fitting it
# don't forget the imputer and the scaler 
pipeline_heart=Pipeline([('imputer',SimpleImputer(strategy='mean')),
                            ('scalar',StandardScaler()),
                            ('model',LogisticRegression())])

grid_values = [{'model': [LogisticRegression(class_weight='balanced', solver = "liblinear")],
                'model__C': np.logspace(-4,4,500),
                'model__penalty': ['l1','l2']},
               {'model': [KNeighborsClassifier()],
                'model__n_neighbors': np.arange(5,505,5),
                'model__weights':['uniform','distance']}]

grid_heart = GridSearchCV(pipeline_heart, 
                         param_grid = grid_values, 
                         scoring="roc_auc",
                         cv = StratifiedKFold(n_splits=5 , shuffle=True, random_state=1234),
                         n_jobs=-1)

grid_heart.fit(X_heart_train, y_heart_train) #train your pipeline

print(f'Grid best score ({grid_heart.scoring}): {grid_heart.best_score_:.4f}')
print(f'Grid best parameter (max.{grid_heart.scoring}): ')
for p,v in grid_heart.best_params_.items():
    print('\t',p,'->',v)



## looking at the best model
from sklearn.metrics import confusion_matrix , recall_score, precision_score
best = grid_heart.best_estimator_

X = X_heart_test
y = y_heart_test

## precision-recall curve
y_scores = best.decision_function(X)#decision_function gives you the proba for a point to be in
prec, rec, thre = precision_recall_curve(y, y_scores)

## confusion matrix
y_pred = best.predict(X)
confusion_m = confusion_matrix(y, y_pred)



fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot( rec, prec  )    
ax[0].set_ylabel('precision')
ax[0].set_xlabel('recall')

sns.heatmap(confusion_m, annot=True , ax = ax[1],fmt='.0f', cmap="crest")
ax[1].set_ylabel('True label')
ax[1].set_xlabel('Predicted label')

print('recall {:.2f}\tprecision {:.2f}'.format(recall_score(y , y_pred) , 
                                               precision_score(y , y_pred)) )


## checking if the model is that bad...
# we do this by training a "dummy" Classifier which gives us a baseline of performance
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
dummy = DummyClassifier()

print("cross-validated average precisions of a dummy classifier:")
print( cross_val_score(dummy, X_heart_train , y_heart_train, cv = 5 , scoring='average_precision') )
print("indeed, this is worse than what our model does.")
print("So, our model does not have crazy performance, but it does bring something to the table")
print("\n***\n")


## checking feature importance to learn more about the problem
from operator import itemgetter
lr = best[-1]
w=lr.coef_[0]#get the weights

featureW = pd.DataFrame( {'feature':df_heart.columns[:-1],'weight':w} )

featureWsorted = featureW.sort_values(by=['weight'] , 
                                      ascending=False , 
                                      key=lambda col : col.abs())

# get the non-null ones
print('Features sorted per importance:')
print( featureWsorted.loc[ featureWsorted["weight"] !=0 ] )
