from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# Creating the object SelectKBest and settling for 5 best features 
skb = SelectKBest(f_classif, k=5)
skb.fit(
    X_cancer, 
    y_cancer)

#get associated pvalues
dico_pval={df_cancer.columns[i]:v for i,v in enumerate(skb.pvalues_)}
sortedPvals = sorted(dico_pval.items(), key=lambda x: x[1], reverse=False) 

print("features F scores (p-values):")
for feature,pval in sortedPvals:
    if pval > 0.01 : # let's ignore everything with a pval>0.01
        print("\t\trest has pval>0.01")
        break 
    print('\t',feature , ':' , pval )

selected5 = [x for x,p in sortedPvals[:5] ]
print("selected best:" , selected10 )


sns.pairplot( df_cancer , hue='malignant' , vars=selected5 )


## that is very nice, but a lot of these are highly correlated...
## Let's start transforming our data so we work with independent features:

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
X_cancer_norm = sc.fit_transform(X_cancer)

pca = PCA()
x_pca = pca.fit_transform(X_cancer_norm)


## now we can select the best feature among the principal components


skb = SelectKBest(f_classif, k=5)
skb.fit(
    x_pca, 
    y_cancer)

#all the features and the chi2 pvalues associated. use .pvalues_
dico_pval={i:v for i,v in enumerate(skb.pvalues_)}
sortedPvals = sorted(dico_pval.items(), key=lambda x: x[1], reverse=False) 

significantComponents = []

print("features Chi2 scores (p-values):")
for feature,pval in sortedPvals:
    if pval > 0.01 : # let's ignore everything with a pval>0.01
        print("\t\trest has pval>0.01")
        break 
    print('\tPC',feature , ':' , pval )
    significantComponents.append(feature)

    
## Wow, they actually correspond to the elements with the highest variance ratio representation!
print( "selected components explained variance fractions:\n\t",pca.explained_variance_ratio_[ significantComponents ] )

print( "Total :\t",sum(pca.explained_variance_ratio_[ significantComponents ]) )

df_pca = pd.DataFrame( x_pca[:, significantComponents ] )
df_pca['target'] = y_cancer

sns.pairplot( df_pca , hue="target" )

df_comp = pd.DataFrame(pca.components_,columns=list(df_cancer.columns)[:-1])
# pca.components_ : recovering the matrix that describe the principal component in the former feature basis. It gives you the 
# values of the coefficients in front of each features to build your PCA components.
plt.figure(figsize=(15,15))
sns.heatmap(df_comp,cmap='coolwarm',cbar_kws={'label':'Eigenvalue'},linewidths=.05)
plt.yticks(np.arange(0+0.5,len(df_comp.columns)+0.5,1),['PCA axis '+str(i) for i in range(len(df_comp.columns))],rotation=0)
plt.show()
