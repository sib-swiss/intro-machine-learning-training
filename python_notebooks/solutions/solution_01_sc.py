df = pd.read_csv('../data/Patel.csv',sep=',', header=0)

df = df.set_index(df.iloc[:,0], inplace=False)
df = df.iloc[:,1:]

X = df.T

print(X.shape)

patients = list(map(lambda s: s[0:5],df.columns)) #take first 5 letter for patient id

color_dict={'MGH26':'blue', 'MGH28':'orange', 'MGH29': 'red', 'MGH30': 'green', 'MGH31': 'pink'}
colors=[color_dict[p] for p in patients]


plt.figure(figsize=(10,10))
ax = sns.heatmap(X, yticklabels=False)
plt.ylabel("Cell type")
plt.xlabel("Gene")
plt.show(block=False)

pca = PCA() #create a PCA object

pca.fit(X)
x_pca = pca.transform(X)

var_explained=pca.explained_variance_ratio_

plt.figure(figsize=(10,10))
plt.scatter(x_pca[:,0],x_pca[:,1],c=colors)
plt.xlabel('First Principal Component ({0:.2f}%)'.format(var_explained[0]*100))
plt.ylabel('Second Principal Component ({0:.2f}%)'.format(var_explained[1]*100))

tsne=TSNE(n_components=2,perplexity=5).fit(X)#create the T-SNE object and fit the data
X_embedded = tsne.embedding_#project the data to the new manifold using the fitted function found before

plt.figure(figsize=(10,10))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, s=60, lw=0)
plt.title('KL divergence {0:.2f}\n perplexity= {1}'.format(tsne.kl_divergence_,5),fontsize=12)
plt.xlabel('First Dimension')
plt.ylabel('Second Dimension')

dbscan = cluster.DBSCAN(eps=20, min_samples=5)
dbscan.fit_predict(X_embedded[:, 0:2])

plt.figure(figsize=(10,10))
idx = np.where(dbscan.labels_>=0)[0]
plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c=[colors[i] for i in idx], s=60, lw=0)
plt.title('KL divergence {0:.2f}\n perplexity= {1}'.format(tsne.kl_divergence_,5),fontsize=12)
plt.xlabel('First Dimension')
plt.ylabel('Second Dimension')


X = X.T

print(X.shape)

kmeans = cluster.KMeans(5)
kmeans.fit(X)
cl_labels = kmeans.labels_

print(Counter(cl_labels))

tsne=TSNE(n_components=2,perplexity=15).fit(X)#create the T-SNE object and fit the data
X_embedded = tsne.embedding_#project the data to the new manifold using the fitted function found before

dbscan = cluster.DBSCAN(eps=2, min_samples=5)
dbscan.fit_predict(X_embedded[:, 0:2])

plt.figure(figsize=(10,10))
idx = np.where(dbscan.labels_>=0)[0]
plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c=[cl_labels[i]  for i in idx], s=60, lw=0, cmap='plasma')
plt.xlabel('First Dimension')
plt.ylabel('Second Dimension')

plt.show()

