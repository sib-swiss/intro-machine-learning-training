#Use the single cell dataset from [Patel et al. Science 2014](https://science.sciencemag.org/content/344/6190/1396).
# It is composed of the expression from different cells in 5 different patients

df_singleCell = pd.read_csv('../data/Patel.csv',sep=',', header=0 , index_col=0)

X_singleCell = df_singleCell.T

print('the data link 430 samples to 5948 genes',X_singleCell.shape)

patients = list(map(lambda s: s[0:5],df_singleCell.columns)) #take first 5 letter for patient id

## here is a color code you may use for your future plots
color_dict={'MGH26':'blue', 'MGH28':'orange', 'MGH29': 'red', 'MGH30': 'green', 'MGH31': 'pink'}
colors=[color_dict[p] for p in patients]

df_singleCell.head()

#So you have a single cell RNA seq from different patient and each patient have different cells.

#This exercise recapitulates different things we have seen until now :

# 1. Plot a heatmap of the data
# 2. Perform and plot a PCA (cell space) plot with patient color code and compare to Figure 1 in the paper
# 3. Use t-SNE (cell space) to vizualize your data to 2 dimensions (perplexity=5), colored according to the patient color code
# 4. Transpose the dataframe (to be on gene space) and perform k-means clustering. Find the optimal number of gene clusters.
# 5. Vizualize the gene clusters in a t-SNE projection on the gene space 
 
#> The matrix `X_singleCell` we provide in the cell above corresponds to "cell x genes", meaning it is good for a cell space projection

#> The last two questions may require a couple of minutes to compute solutions. 

#1. Plot a heatmap of the data
sns.heatmap(X_singleCell)

# 2. Perform and plot a PCA (cell space) plot with patient color code and compare to Figure 1 in the paper

## PCA
scellPCA = PCA()
scellPCA.fit(X_singleCell)
x_sc_pca = scellPCA.transform(X_singleCell)

## plotting - here using seaborn hue argument helps us get a nice legend easily
sns.scatterplot( x=x_sc_pca[:,0] , y=x_sc_pca[:,1], 
                hue=patients , palette = color_dict , s=60)

var_explained = scellPCA.explained_variance_ratio_

plt.xlabel('First Principal Component ({0:.2f}%)'.format(var_explained[0]*100))
plt.ylabel('Second Principal Component ({0:.2f}%)'.format(var_explained[1]*100))

# 3. Use t-SNE (cell space) to vizualize your data to 2 dimensions (perplexity=5), colored according to the patient color code
tsne_exo_sample=TSNE(n_components=2,perplexity=20).fit(X_singleCell)
X_embedded_exo_sample = tsne_exo_sample.embedding_

plt.figure(figsize=(10,10))
plt.title('a point is a sample',fontsize=20)
sns.scatterplot(X_embedded_exo_sample[:, 0], 
                X_embedded_exo_sample[:, 1], 
                hue=patients , palette = color_dict, s=60, lw=0)
plt.xlabel('First Dimension')
plt.ylabel('Second Dimension')


# 4. Transpose the dataframe (to be on gene space) and perform k-means clustering (5 for 5 patients)
X_sc_genes = X_singleCell.T

pca_sc_genes = PCA() #create a PCA object

pca_sc_genes.fit(X_sc_genes)
x_pca_sc_genes = pca_sc_genes.transform(X_sc_genes)


inertias = []
silhouettes = []

nb_clusters = np.arange(2,17)

for k in nb_clusters: # testing a number of clusters from 2 to 16
    kmeans = cluster.KMeans(k)
    kmeans.fit(x_pca_sc_genes)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(x_pca_sc_genes,kmeans.labels_))

plt.subplots(figsize=(15,7))

plt.subplot(1,2,1)
plt.plot(nb_clusters, inertias, ls="-", lw=2)
plt.xlabel('Number of clusters',fontsize=20)
plt.ylabel('Inertia',fontsize=20)
plt.title('k-means clustering',fontsize=20)

plt.subplot(1,2,2)
plt.plot(nb_clusters, silhouettes, ls="-", lw=2)
plt.xlabel('Number of clusters',fontsize=20)
plt.ylabel('Silhouette  score',fontsize=20)
plt.title('k-means clustering',fontsize=20)

plt.tight_layout()
plt.show()

bestI = np.argmax( silhouettes )
bestK = nb_clusters[bestI]

print('best number of clusters',bestK)

# 5. Vizualize the gene clusters in a t-SNE projection on the gene space 
from collections import Counter

# we perform the Kmean that gave to best silhouette score
kmeans_sc_genes = cluster.KMeans(bestK)
kmeans_sc_genes.fit(x_pca_sc_genes)

# reporting the number of observations for each clusters
cluster_labels_sc_genes = kmeans_sc_genes.labels_
print(Counter(cluster_labels_sc_genes))

## t-SNE embedding of the gene space data
tsne_sc_genes=TSNE(n_components=2,perplexity=30).fit(X_sc_genes)
X_embedded_sc_genes = tsne_sc_genes.embedding_


plt.figure(figsize=(10,10))
plt.title('t-SNE - a point is a gene',fontsize=20)
plt.scatter(X_embedded_sc_genes[:, 0], 
            X_embedded_sc_genes[:, 1], 
            c=cluster_labels_sc_genes, s=60, lw=0, cmap='plasma')
plt.xlabel('First Dimension')
plt.ylabel('Second Dimension')
plt.xlim([-5,5])
plt.ylim([-5,5])

plt.show()