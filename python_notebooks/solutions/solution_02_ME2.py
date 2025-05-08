feature_names = list( df_cancer.columns) [:-1]

logCs = []

coef_dict = {'name' : [],
             'val' : [],
             'log_C' : []}
accuracies = []

for C in np.logspace(-4,1,100):

    lr = LogisticRegression( penalty = 'l2' , C = C  , solver='liblinear')
    lr.fit(X_train_norm , y_train)
    
    logCs.append(np.log10(C))
    accuracies.append( accuracy_score( y_valid , lr.predict(X_valid_norm) ) )
    
    coef_dict['name'] += list( feature_names )
    coef_dict['val'] += list( lr.coef_[0] )
    coef_dict['log_C'] += [np.log10(C)]* len(feature_names )

coef_df = pd.DataFrame(coef_dict)
bestC = logCs[ np.argmax( accuracies ) ]

fig,ax = plt.subplots(1,2,figsize = (20,10))

ax[0].plot(logCs , accuracies)
ax[0].set_xlabel("log10( C )")
ax[0].set_ylabel("validation accuracy")
ax[0].axvline( bestC, color='r', ls = '--' )

sns.lineplot( x = 'log_C' , y='val' , hue = 'name' , data= coef_df , ax = ax[1])
ax[1].axvline( bestC , color='r', ls = '--' )

fig.suptitle("logistic regression of cancer data with an L2 regularization.")
