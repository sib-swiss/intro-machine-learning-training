## 1. handle the NAs. A mean imputation should work here

#handling NAs
df_mam_I = df_mam.fillna(df_mam.mean())


## 2. perform a PCA. Plot the PCA projected data as well as the weight of each column on the axes. What can you say ?
sc = StandardScaler()
df_mam_IS = sc.fit_transform(df_mam_I)

pca_mam = PCA()

x_pca = pca_mam.fit_transform(df_mam_IS)

plt.subplots(figsize=(15,15))

## plotting 
sns.scatterplot( x=x_pca[:,0] , y=x_pca[:,1] , s=0)
## adding the species name 
for i,sp in enumerate( df_mam.index ):
    plt.text( x_pca[i,0] , x_pca[i,1] , sp , color='blue' )


var_explained = pca_mam.explained_variance_ratio_

plt.xlabel('First Principal Component ({0:.2f}%)'.format(var_explained[0]*100))
plt.ylabel('Second Principal Component ({0:.2f}%)'.format(var_explained[1]*100))

feature_vectors = pca_mam.components_.T
arrow_size = 10

# projections of the original features
for i, v in enumerate(feature_vectors): # enumerate over the rows of feature_vectors
    plt.arrow(0, 0, arrow_size * v[0], arrow_size * v[1], 
              head_width=0.00008, head_length=0.00008, width=0.00005, color='k')
    
    text_pos = -0.005 if v[0] < 0 else 0.0001
    
    plt.text(v[0]*arrow_size+text_pos, 
             v[1]*arrow_size+text_pos, 
             df_mam.columns[i] ,fontsize=10)
    

## 3. use t-SNE to get an embedding of the data in 2D and represent it.
##     **bonus :** plot the species names in the embedded space with `plt.text`

tsne_exo_sample=TSNE(n_components=2,perplexity=20).fit(df_mam_IS)
X_embedded_exo_sample = tsne_exo_sample.embedding_

plt.figure(figsize=(10,10))
plt.title('a point is a sample',fontsize=20)
sns.scatterplot(X_embedded_exo_sample[:, 0], 
                X_embedded_exo_sample[:, 1],  s=0, lw=0)
for i,sp in enumerate( df_mam.index ):
    plt.text( X_embedded_exo_sample[i,0] , X_embedded_exo_sample[i,1] , sp , color='blue' )


plt.xlabel('First Dimension')
plt.ylabel('Second Dimension')


## 4. perform a Kmean clustering on the PCA projected data. What is the best number of cluster according to the silhouette score?
nr_clusters = np.arange(15)+2
inertias , silhouettes = getSilhouetteProfile( x_pca , nr_clusters )

## getting the K with maximum silhouette
bestI = np.argmax( silhouettes )
bestK = nr_clusters[bestI]
print('best K :',bestK)

plt.subplots(figsize=(15,7))

plt.subplot(1,2,1)
plt.plot(nr_clusters, inertias, ls="-", lw=2)
plt.xlabel('Number of clusters',fontsize=20)
plt.ylabel('Inertia',fontsize=20)
plt.title('k-means clustering',fontsize=20)

plt.subplot(1,2,2)
plt.plot(nr_clusters, silhouettes, ls="-", lw=2)
plt.xlabel('Number of clusters',fontsize=20)
plt.ylabel('Silhouette  score',fontsize=20)
plt.title('k-means clustering',fontsize=20)

plt.tight_layout()
plt.show()


## 5. plot the t-SNE projected data colored according to the cluster they belong to.
kmeans_mam = cluster.KMeans(bestK)
kmeans_mam.fit(x_pca)

# reporting the number of observations for each clusters
cluster_labels_mam = kmeans_mam.labels_
## I put here a bunch of colors I like
cluster_2_colors = [ 'xkcd:teal' , 'xkcd:lavender' , 'xkcd:mustard' , 'xkcd:sage' ]

print(Counter(cluster_labels_mam))

plt.figure(figsize=(10,10))
plt.title('a point is a sample',fontsize=20)
sns.scatterplot(X_embedded_exo_sample[:, 0], 
                X_embedded_exo_sample[:, 1],  s=0, lw=0)
for i,sp in enumerate( df_mam.index ):
    plt.text( X_embedded_exo_sample[i,0] , X_embedded_exo_sample[i,1] , sp , 
             color= cluster_2_colors[cluster_labels_mam[i]] )


plt.xlabel('First Dimension')
plt.ylabel('Second Dimension')
