# 1. perform a PCA and vizualize the results
from sklearn.decomposition import PCA

pca = PCA()
X_rpkm_pca = pca.fit_transform(X_rpkm)

## plotting the results
fig, ax = plt.subplots(1,2,figsize=(10,6))
ax[0].plot( np.log10(pca.explained_variance_ratio_ ))
sns.scatterplot( x = X_rpkm_pca[:,0],
                 y = X_rpkm_pca[:,1] , ax=ax[1] , color= index_colors)
ax[1].set_xlabel("PC 0 - {:.1f}% variance".format(pca.explained_variance_ratio_[0]*100))
ax[1].set_ylabel("PC 1 - {:.1f}% variance".format(pca.explained_variance_ratio_[1]*100))
fig.tight_layout()
## fancier representation with 3D scatterplot
import plotly.express as px
import plotly.graph_objects as go

fig = px.scatter_3d( x = X_rpkm_pca[:,0] , y = X_rpkm_pca[:,1], z = X_rpkm_pca[:,2], 
                    color = index_colors,
                    color_discrete_map = 'identity' )
fig
# 2. try to cluster the cells using K-Means or Hierarchical clustering. 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score , adjusted_rand_score
## I will first do a single Kmean with K=7

km = KMeans(n_clusters=7)
km.fit(X_rpkm_pca)

silhouette = silhouette_score( X_rpkm_pca , km.labels_ ) 
ARI = adjusted_rand_score( km.labels_ , index_colors )

sns.scatterplot( x = X_rpkm_pca[:,0],
                 y = X_rpkm_pca[:,1] , hue = km.labels_.astype(str))

plt.title( f"Kmeans K=7 - silhouette score {silhouette:.2f} - ARI {ARI:.2f}" )
#     Try several parameter values and evaluate your clustering with a metric.

fig,ax = plt.subplots( 1 , 2 , figsize = (15,8) )

## Kmeans ##
silhouettes , ARIs , Ks = [] , [] , []

for K in range(2,21) :
    km = KMeans(n_clusters=K)
    km.fit(X_rpkm_pca)
    silhouettes.append(  silhouette_score(X_rpkm_pca,km.labels_ ) )
    ARIs.append( adjusted_rand_score( km.labels_ , index_colors ) )
    Ks.append( K )
ax[0].plot( Ks, silhouettes , label = 'K-means' )
ax[1].plot( Ks, ARIs , label = 'K-means' )

## hierarchical clustering - single and average ##
for link in ['single','average']:
    silhouettes , ARIs , Ks = [] , [] , []

    for K in range(2,21) :
        hclust = AgglomerativeClustering(n_clusters = K , linkage = link )
        hclust.fit(X_rpkm_pca)
        silhouettes.append(  silhouette_score( X_rpkm_pca , hclust.labels_ ) )
        ARIs.append( adjusted_rand_score( hclust.labels_ , index_colors ) )
        Ks.append( K )
    ax[0].plot( Ks, silhouettes , label = f'Hierarchical - {link}' )
    ax[1].plot( Ks, ARIs , label = f'Hierarchical - {link}' )


ax[0].set_ylabel('silhouette')
ax[0].set_xlabel('number of clusters')
ax[0].set_xticks(Ks)
ax[0].grid(axis='x')
ax[0].legend()
ax[1].set_ylabel('ARI')
ax[1].set_xlabel('number of clusters')
ax[1].set_xticks(Ks)
ax[1].grid(axis='x')
ax[1].legend()
fig.tight_layout()
## best clustering according to the Silhouette criterion
km = KMeans(2)
km.fit(X_rpkm_pca)
sns.scatterplot( x = X_rpkm_pca[:,0],
                 y = X_rpkm_pca[:,1] , hue = km.labels_.astype(str) )
# 3. try the same thing, but use only the 50 first PCA components. Do you get better results?

data = X_rpkm_pca[:,:50]

fig,ax = plt.subplots( 1 , 2 , figsize = (15,8) )

## Kmeans ##
silhouettes , ARIs , Ks = [] , [] , []

for K in range(2,21) :
    km = KMeans(n_clusters=K)
    km.fit(data)
    silhouettes.append(  silhouette_score(data,km.labels_ ) )
    ARIs.append( adjusted_rand_score( km.labels_ , index_colors ) )
    Ks.append( K )
ax[0].plot( Ks, silhouettes , label = 'K-means' )
ax[1].plot( Ks, ARIs , label = 'K-means' )

## hierarchical clustering - single and average ##
for link in ['single','average']:
    silhouettes , ARIs , Ks = [] , [] , []

    for K in range(2,21) :
        hclust = AgglomerativeClustering(n_clusters = K , linkage = link )
        hclust.fit(data)
        silhouettes.append(  silhouette_score( data , hclust.labels_ ) )
        ARIs.append( adjusted_rand_score( hclust.labels_ , index_colors ) )
        Ks.append( K )
    ax[0].plot( Ks, silhouettes , label = f'Hierarchical - {link}' )
    ax[1].plot( Ks, ARIs , label = f'Hierarchical - {link}' )


ax[0].set_ylabel('silhouette')
ax[0].set_xlabel('number of clusters')
ax[0].set_xticks(Ks)
ax[0].grid(axis='x')
ax[0].legend()
ax[1].set_ylabel('ARI')
ax[1].set_xlabel('number of clusters')
ax[1].set_xticks(Ks)
ax[1].grid(axis='x')
ax[1].legend()
fig.tight_layout()
