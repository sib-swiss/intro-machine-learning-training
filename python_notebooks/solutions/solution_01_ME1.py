fig,ax = plt.subplots( 1,2,figsize=(15,7))

sns.heatmap(X_ph_imputed.corr(),ax=ax[0], cmap='jet' , vmin = 0 , vmax = 1)
ax[0].set_title("half-min imputed data")

sns.heatmap(X_ph_positive.corr(), ax= ax[1], cmap='jet' , vmin = 0 , vmax = 1)
ax[1].set_title("removed missing data")

fig.suptitle('sample correlation')
plt.tight_layout()
