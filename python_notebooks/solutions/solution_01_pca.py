import seaborn as sns

sns.set_context("paper", rc={"axes.labelsize":20})
pca = PCA()

## we fit and transform the PCA on the gene expression data only (all columns except the last one).
x_pca = pca.fit_transform(X_aml) # calculates coordinates of row vectors in X in PC space

## we extract the components weights
feature_vectors = pca.components_.T

## plotting 
arrow_size = 10

fig,axes = plt.subplots(1,2,figsize=(14,10))

# left subplot : %explained variance for each axis
axes[0].scatter(np.arange(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_,s=80,c='k')
axes[0].set_ylabel('Fraction of variance explained')
axes[0].set_xlabel('PC rank')


# right subplot : PC0 and PC1, colored by auc

aa=axes[1].scatter(x_pca[:,0],x_pca[:,1],c=df_aml.auc,s=80,cmap='plasma')
axes[1].set_xlabel('First Principal Component ({0:.2f}%)'.format(pca.explained_variance_ratio_[0]*100))
axes[1].set_ylabel('Second Principal Component ({0:.2f}%)'.format(pca.explained_variance_ratio_[1]*100))

cmap = plt.get_cmap('gnuplot')
n = feature_vectors.shape[0]
for i, v in enumerate(feature_vectors): # enumerate over the rows of feature_vectors
    axes[1].arrow(0, 0, arrow_size * v[0], arrow_size * v[1], head_width=0.00008, head_length=0.00008, width=0.00005,
              color=cmap((1.0 * i) / n))
    text_pos = -0.005 if v[0] < 0 else 0.0001
    axes[1].text(v[0]*arrow_size+text_pos, v[1]*arrow_size+0.0001, X_aml.columns[i], 
            color=cmap((1.0 * i) / n),fontsize=10)


axo = fig.add_axes([0.95,0.2,0.01,0.5])

axes[1].set_title("PCA projection colored by AUC",fontsize=25)
fig.colorbar(aa, ax =axes[1] , cax=axo, orientation='vertical' , fraction=0.01)


