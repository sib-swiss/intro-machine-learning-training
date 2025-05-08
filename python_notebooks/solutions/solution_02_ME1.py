from sklearn.metrics import silhouette_score

silhouettes = []

Ks = range( 2,10 )
for i,K in enumerate( Ks ):
    km = KMeans(n_clusters = K)
    km.fit( X_aml_pca ) 
    
    ## computing silhouette
    silhouettes.append( silhouette_score( X_aml_pca , km.labels_ )    )
    
# fitting Kmean with the K giving the best silhouette score
bestK = Ks[ np.argmax(silhouettes) ]
km = KMeans(n_clusters = bestK)
km.fit( X_aml_pca ) 


fig,ax = plt.subplots( 1,3 , figsize = (15,6) )
ax[0].plot( Ks, silhouettes )
ax[0].set_ylabel('silhouette')
ax[0].set_xlabel('K')
ax[0].grid(axis = 'x')


sns.scatterplot( x = X_aml_pca[:,0] , y = X_aml_pca[:,1], hue = km.labels_, ax = ax[1])
ax[1].set_xlabel('First Principal Component ({0:.2f}%)'.format(pca.explained_variance_ratio_[0]*100))
ax[1].set_ylabel('Second Principal Component ({0:.2f}%)'.format(pca.explained_variance_ratio_[1]*100))

sns.boxplot( y = df_aml.auc , x = km.labels_ , fliersize=0 , fill = False, ax = ax[2])
sns.stripplot( y = df_aml.auc , x = km.labels_, dodge=True, color = 'grey', ax = ax[2])
ax[2].set_xlabel('K-mean labels')

plt.tight_layout()
