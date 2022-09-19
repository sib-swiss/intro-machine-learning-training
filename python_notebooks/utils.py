import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn import tree
import collections
from IPython.display import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier

plt.rc('xtick', color='k', labelsize='medium', direction='in')
plt.rc('xtick.major', size=8, pad=12)
plt.rc('xtick.minor', size=8, pad=12)

plt.rc('ytick', color='k', labelsize='medium', direction='in')
plt.rc('ytick.major', size=8, pad=12)
plt.rc('ytick.minor', size=8, pad=12)

def make_meshgrid(x, y, n=100):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    n: number of intermediary points (optional)

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    #xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                     np.arange(y_min, y_max, h))
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n),
                         np.linspace(y_min, y_max, n))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def contour_knn(n,X,y,w , resolution = 100, ax = None):
    '''
        Takes:
            - n : number of nearest neighbors
            - X : feature matrix
            - y : label
            - w : voting rule
            - resolution = 100 : number of points in the mesh (high number affect performance)
            - ax = None : ax for plotting

        returns:
            (matplotlib.Axe)
    '''
    models = KNeighborsClassifier(n_neighbors = n,weights=w, n_jobs=-1)
    models = models.fit(X, y) 

        # title for the plots
    titles = 'K neighbors k='+str(n)+', '+w


    if ax is None:
        fig, ax = plt.subplots(1, 1)
        

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1 , n=resolution)



    plot_contours(ax, models, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(titles)
    return ax

def makeROCcurveBin( X,y,model , ax ):
    """
        Takes:
            * p : penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
            * X : covariables
            * y : target
            * c : inverse regularization strength
    """
    y_score_logi_r_c = model.decision_function(X)
    fpr_logi_r_c, tpr_logi_r_c, thre = roc_curve(y, y_score_logi_r_c)
    roc_auc_logi_r_c = auc(fpr_logi_r_c, tpr_logi_r_c)
    score=model.score(X,y)


    ax.set_xlim([-0.01, 1.00])
    ax.set_ylim([-0.01, 1.01])
    ax.plot(fpr_logi_r_c, tpr_logi_r_c, lw=3, label='LogRegr ROC curve\n (area = {:0.2f})\n Acc={:1.3f}'.format(roc_auc_logi_r_c,score))
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_title('ROC curve (logistic classifier)', fontsize=16)
    ax.legend(loc='lower right', fontsize=13)
    ax.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    return 

def makeROCcurveMulti( X,y,model , ax ):
    """
        Takes:
            * p : penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
            * X : covariables
            * y : target
            * c : inverse regularization strength
    """
    n_classes = len(set(y))
    y = label_binarize(y, classes=np.arange(0,n_classes,1))
    classifier = OneVsRestClassifier(model)
    y_score = classifier.fit(X, y).decision_function(X)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 3
    # Plot all ROC curves
    
    

    ax.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multi class Receiver operating characteristic curve')
    ax.legend(loc="lower right")

    return 

def makeROCcurve(X,y,model,ax):
    
    nb_classes = len(set(y))
    if nb_classes > 2:
        makeROCcurveMulti(X,y,model,ax)
    else:
        makeROCcurveBin(X,y,model,ax)

def contour_lr(p,X,y,c,mult='ovr'):
    """
        Takes:
            * p : penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
            * X : covariables
            * y : target
            * c : inverse regularization strength
            * mult = 'ovr': how to handle multi-class {‘auto’, ‘ovr’, ‘multinomial’}
    """
    models = LogisticRegression(penalty = p,C=c, multi_class=mult)
    # Create the logistic regresison object(with 3 main hyperparameters!!)
    # penalty is either l1 or l2, C is how much weight we put on the regularization, multi_calss is how we proceed when multiclasses
    models = models.fit(X, y)
    dico_color={0:'blue',1:'white',2:'red'}

    titles = 'Logistic regression penalty='+str(p)+' C='+str(c)

    nbCategories = len( set(y) )
    if nbCategories> 3 :
        print("more than 3 categories detected... problem may occur with this code")
    
    ## simple mesh
    #nl = int( np.ceil(nbCategories/2) ) # number of category lines
    #fig, ax = plt.subplots(nl+1,2,figsize=(10,5 + 2.5*nl),
    #                      gridspec_kw = {'height_ratios':[2]+[1]*(nl)})
    fig, ax = plt.subplots(1,2,figsize=(10,5))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax[0], models, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax[0].scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    interc=models.intercept_
    wei=models.coef_
    for i in range(len(interc)):
        ax[0].plot([xx.min(),xx.max()],
                      [-(interc[i]+wei[i][0]*xx.min())/wei[i][1],-(interc[i]+wei[i][0]*xx.max())/wei[i][1]],
                 color=dico_color[i],ls='--')
    ax[0].set_xlim(xx.min(), xx.max())
    ax[0].set_ylim(yy.min(), yy.max())
    ax[0].set_xticks(())
    ax[0].set_yticks(())
    ax[0].set_title(titles)

   
    ## plotting decision functions
    xx = np.linspace(np.min(X0)-5, np.max(X0)+5, 100)
    yy = np.linspace(np.min(X1)-5, np.max(X1)+5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    
    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    #print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

    ## plotting ROCcurve
    makeROCcurve(X,y,models,ax[1])
    
    plt.tight_layout()
    
    
    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    
    fig, ax = plt.subplots(1,n_classes,figsize=(10,5))
    
    for k in range(n_classes):
        if n_classes==1:
            AX=ax
        else:
            AX = ax[k]
        
        
        
        imshow_handle = AX.imshow(probas[:, k].reshape((100, 100)),extent=(np.min(X0)-5, np.max(X0)+5, np.min(X1)-5, np.max(X1)+5), origin='lower',cmap='plasma')
        AX.set_xticks(())
        AX.set_xlim([np.min(X0)-5, np.max(X0)+5])
        AX.set_ylim([np.min(X1)-5, np.max(X1)+5])
        AX.set_yticks(())
        AX.set_title('Class '+str(k),fontsize=25)
        for i in range(len(interc)):
            AX.plot([np.min(X0)-5,np.max(X0)+5],
                    [-(interc[i]+wei[i][0]*(np.min(X0)-5))/wei[i][1],-(interc[i]+wei[i][0]*(np.max(X0)+5))/wei[i][1]],
                 color=dico_color[i],ls='--')
        idx = (y_pred == k)
        
        if idx.any():
            
            AX.scatter(X[idx, 0], X[idx, 1],
                       s = 100,marker='o', 
                       c=[dico_color[h] for h in y[idx]], 
                       edgecolor='k')

        #cbar=plt.colorbar(imshow_handle,ax=axs[k])
        #cbar.ax.set_title('Probability',fontsize=10)
    
    
    plt.tight_layout()
    axo = plt.axes([0,0,1,0.05])

    plt.colorbar(imshow_handle, cax=axo, orientation='horizontal')
    axo.set_xlabel("Probability",fontsize=25)
    

    

def roc_multi_ovr(grid_lr_acc_i,
                  n_classes,
                  X_train,
                  y_train,
                  X_test,
                  y_test):
    L = list(set(y_test))
    L.sort()
    y = label_binarize(y_test, classes=L)
    
    classifier = OneVsRestClassifier(LogisticRegression(penalty = grid_lr_acc_i.best_params_['model__penalty'],
                                                        C=grid_lr_acc_i.best_params_['model__C'],solver='liblinear'))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for j in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[j], tpr[j])

            # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 3
    # Plot all ROC curves
    plt.figure(figsize=(7,7))
    plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.3f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.3f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.3f})'
                         ''.format(L[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi class Receiver operating characteristic curve\nOnevsRest')
    plt.legend(loc="lower right")
    plt.show()

#This you don't care too much either but if you want to use it one day it is here : It is a way to plot Roc curves in the 
#context of one vs one  multiclass
def roc_multi_ovo(grid_lr_acc_i,n_classes,X_train,y_train,X_test,y_test):

    n_classes=3
    y_list=[]
    for i in range(n_classes):
        glen=[]
        for j in range(i+1,n_classes):
            glen.append(label_binarize(np.array(y_test), classes=[i,j]))

        if len(glen)>0:
            y_list.append(np.concatenate(glen))

    y=np.vstack(y_list)

    y = label_binarize(y_test, classes=np.arange(0,n_classes,1))
    classifier = OneVsOneClassifier(LogisticRegression(penalty = grid_lr_acc_i.best_params_['model__penalty'],
                                                        C=grid_lr_acc_i.best_params_['model__C'],solver='liblinear'))
    #y_p=classifier.fit(X_test, y_test).predict(X_test)
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    



    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    k=0
    for i in range(n_classes):
        for j in range(i+1,n_classes):
            fpr[str(i)+'_'+str(j)], tpr[str(i)+'_'+str(j)], _ = roc_curve(y[:, k], y_score[:, k])
            roc_auc[str(i)+'_'+str(j)] = auc(fpr[str(i)+'_'+str(j)], tpr[str(i)+'_'+str(j)])
            k+=1

        # Compute micro-average ROC curve and ROC area

    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr.keys()]))

        # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for j in fpr.keys():
        if j!='micro':
            mean_tpr += interp(all_fpr, fpr[j], tpr[j])

            # Finally average it and compute AUC
    mean_tpr /= (n_classes*(n_classes-1)/2)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 3
    # Plot all ROC curves
    plt.figure(figsize=(7,7))
    plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(fpr.keys(), colors):
        if i!="macro" and i!="micro":
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(i, roc_auc[i]))


    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
            #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.title('Multi class Receiver operating characteristic curve\nOnevsOne')
    plt.legend(loc="lower right")
    plt.show()

def contour_SVM(X,y,c,ker,deg=2,gam=1,mult='ovr'):
    """
    Takes:
        * X : covariable 
        * y : target
        * c : regulatization parameter
        * ker : kernel
        * deg : degree
        * gam : gamma
        * mult : decision function shape
    """
    models = svm.SVC(C=c, kernel=ker, degree=deg, gamma= gam, decision_function_shape=mult,probability=True)
    #those are all the hyperparameters that are, in my opinion, important to tune. C is again the good old inverse of the weight for l2 
    #regularization, kernel is the dot product you want to use, degree is the degree of the polynomial kernel you want to use,
    #gamma is the standard deviation for the Gaussian Radial Basis function, decision_function_shape is used in case of multiclass,
    #proba = True is just here so we can draw the proba contour in our plot.
    models = models.fit(X, y)
    dico_color={0:'blue',1:'white',2:'red'}

    titles = 'SVM'+' C='+str(c)+' '+ker 
    
    fig, ax = plt.subplots(1,2,figsize=(10,5))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax[0], models, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax[0].scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    if ker=='linear':
        interc=models.intercept_
        wei=models.coef_
        for i in range(len(interc)):
            ax[0].plot([xx.min(),xx.max()],
                     [-(interc[i]+wei[i][0]*xx.min())/wei[i][1],-(interc[i]+wei[i][0]*xx.max())/wei[i][1]],
                 color=dico_color[i],ls='--')
    ax[0].set_xlim(xx.min(), xx.max())
    ax[0].set_ylim(yy.min(), yy.max())
    ax[0].set_xticks(())
    ax[0].set_yticks(())
    ax[0].set_title(titles)

   
    ## plotting decision functions
    xx = np.linspace(np.min(X0)-5, np.max(X0)+5, 100)
    yy = np.linspace(np.min(X1)-5, np.max(X1)+5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    
    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    #print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

    ## plotting ROCcurve
    makeROCcurve(X,y,models,ax[1])
    
    plt.tight_layout()
    
    
    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    
    fig, ax = plt.subplots(1,n_classes,figsize=(10,5))
    
    for k in range(n_classes):
        if n_classes==1:
            AX=ax
        else:
            AX = ax[k]
        
        
        
        imshow_handle = AX.imshow(probas[:, k].reshape((100, 100)),extent=(np.min(X0)-5, np.max(X0)+5, np.min(X1)-5, np.max(X1)+5), origin='lower',cmap='plasma')
        AX.set_xticks(())
        AX.set_xlim([np.min(X0)-5, np.max(X0)+5])
        AX.set_ylim([np.min(X1)-5, np.max(X1)+5])
        AX.set_yticks(())
        AX.set_title('Class '+str(k),fontsize=25)
        if ker == "linear":
            for i in range(len(interc)):
                AX.plot([np.min(X0)-5,np.max(X0)+5],
                        [-(interc[i]+wei[i][0]*(np.min(X0)-5))/wei[i][1],-(interc[i]+wei[i][0]*(np.max(X0)+5))/wei[i][1]],
                 color=dico_color[i],ls='--')
        idx = (y_pred == k)
        
        if idx.any():
            
            AX.scatter(X[idx, 0], X[idx, 1],
                       s = 100,marker='o', 
                       c=[dico_color[h] for h in y[idx]], 
                       edgecolor='k')

        #cbar=plt.colorbar(imshow_handle,ax=axs[k])
        #cbar.ax.set_title('Probability',fontsize=10)
    
    
    plt.tight_layout()
    axo = plt.axes([0,0,1,0.05])

    plt.colorbar(imshow_handle, cax=axo, orientation='horizontal')
    axo.set_xlabel("Probability",fontsize=25)
    



def contour_tree(X,y,crit,maxd,min_s,min_l,max_f):#to understand what those hyperparameters stand for just check the first example
    models = DecisionTreeClassifier(criterion=crit,max_depth=maxd,min_samples_split=min_s,min_samples_leaf=min_l,max_features=max_f)
    models = models.fit(X, y) 

        # title for the plots
    titles = 'Decision tree '+' '.join([str(crit),str(maxd),str(min_s),str(min_l),str(max_f)])

        # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots(1, 1)
        #plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)



    plot_contours(ax, models, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_xticks(())
    #ax.set_yticks(())
    ax.set_title(titles)
        
    plt.show()
    
    dot_data = tree.export_graphviz(models,
                                feature_names=['x','y'],
                                out_file=None,
                                filled=True,
                                rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    return Image(graph.create_png())

def contour_RF(X,y,n_tree,crit,maxd,min_s,min_l,max_f):
    """
    Performs a classification using a random forest and plots a 2D decision space
    and then does the same for a single tree classifier with similar hyper parameters for comparison
    
    Takes:
        * X : covariables
        * y : target
        * n_tree : number of tree in the forest
        * crit : impurity criterion
        * maxd : tree max depth
        * min_s : minimum number of samples to consider an internal node rule
        * min_l : minimum number of samples to consider an leaf node rule
        * max_f : maximum number of features to consider at a node
    """
    
    models = RandomForestClassifier(n_tree,criterion=crit,max_depth=maxd,min_samples_split=min_s,min_samples_leaf=min_l,max_features=max_f)
    models = models.fit(X, y) 
    dico_color={0:'blue',1:'white',2:'red'}
        # title for the plots
    titles = 'Random Forest '+' '.join([str(crit),
                                        str(maxd),
                                        str(min_s),
                                        str(min_l),
                                        str(max_f)])


    nCat = len(set(y))
    
    fig = plt.figure(constrained_layout=True,figsize=(10,4+np.ceil(nCat/4)*4))
    gs = GridSpec( 2+ int(np.ceil(nCat/4)), 4, figure=fig)
    #print( 2+ int(np.ceil(nCat/4)), 4 )

    ### plot 1 : RF contour
    ax = fig.add_subplot(gs[:2, :2])
    
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    Xfull = np.c_[xx.ravel(), yy.ravel()]


    plot_contours(ax, models, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(titles)
    
    ## probability contour for each category
    xx = np.linspace(np.min(X0)-5, np.max(X0)+5, 100)
    yy = np.linspace(np.min(X1)-5, np.max(X1)+5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    
    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    

    for k in range(n_classes):
        #print(k,2+k//4, k%4)
        ax = fig.add_subplot(gs[2+k//4, k%4])
        
        if k == 0:
            ax.set_ylabel('Random Forest')
        imshow_handle = ax.imshow(probas[:, k].reshape((100, 100)),
                                  extent=(np.min(X0)-5, np.max(X0)+5, 
                                          np.min(X1)-5, np.max(X1)+5), 
                                  origin='lower',cmap='plasma')
        ax.set_xticks(())
        ax.set_xlim([np.min(X0)-5, np.max(X0)+5])
        ax.set_ylim([np.min(X1)-5, np.max(X1)+5])
        ax.set_yticks(())
        ax.set_title('Class '+str(k),fontsize=25)
        
        idx = (y_pred == k)
        
        if idx.any():
            
            ax.scatter(X[idx, 0], X[idx, 1],
                       s=100, marker='o', 
                       c=[dico_color[h] for h in y[idx]], edgecolor='k')

    
    ## comparing with a decision tree
    models = DecisionTreeClassifier(criterion=crit,
                                    max_depth=maxd,
                                    min_samples_split=min_s,
                                    min_samples_leaf=min_l,
                                    max_features=max_f)
    models = models.fit(X, y) 

        # title for the plots
    titles = 'Decision tree '+' '.join([str(crit),str(maxd),str(min_s),str(min_l),str(max_f)])

    ### plot 1 : RF contour
    ax = fig.add_subplot(gs[:2, 2:])

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    plot_contours(ax, models, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(titles)
    plt.show()
    
    ax = plt.axes([0,0,1,0.05])
    plt.title("Probability",fontsize=25)
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
    plt.show()

    
def contour_ADA(X,y,n_estimators,learning_rate):
    '''
    Takes:
        * X : covariables
        * y : target
        * n_estimators : number of stumps
        * learning_rate : learning rate
    
    '''
    n_tree=n_estimators
    learn_r=learning_rate
    models = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
    models = models.fit(X, y) 
    dico_color={0:'blue',1:'white',2:'red'}
        # title for the plots
    titles = 'Adaboost '+' '.join([str(n_tree),str(learn_r)])

        # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots(1, 1,figsize=(5,5))
        #plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    
    plot_contours(ax, models, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_xticks(())
    #ax.set_yticks(())
    ax.set_title(titles)
        #plt.savefig('C:\\Users\\sebas\\Desktop\\cours_scikit-learn\\Iris_example_knn_1_'+str(i)+'.pdf')
    plt.show()
    
    xx = np.linspace(np.min(X0)-5, np.max(X0)+5, 100)
    yy = np.linspace(np.min(X1)-5, np.max(X1)+5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    
    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    
    plt.figure(figsize=(10*n_classes,10))
    for k in range(n_classes):
        plt.subplot(1, n_classes, k + 1)
        #plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel('Adaboost')
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),extent=(np.min(X0)-5, np.max(X0)+5, np.min(X1)-5, np.max(X1)+5), origin='lower',cmap='plasma')
        plt.xticks(())
        plt.xlim([np.min(X0)-5, np.max(X0)+5])
        plt.ylim([np.min(X1)-5, np.max(X1)+5])
        plt.yticks(())
        plt.title('Class '+str(k),fontsize=25)
        
        idx = (y_pred == k)
        
        if idx.any():
            
            plt.scatter(X[idx, 0], X[idx, 1],s=100, marker='o', c=[dico_color[h] for h in y[idx]], edgecolor='k')

    ax = plt.axes([0,0,1,0.05])
    plt.title("Probability",fontsize=25)
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

    plt.show()


def contour_BG(X,y,
                n_estimators,
                learning_rate,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features='auto'):
    """
    Takes: 
        * X : covariables data
        * y : target
        * n_estimators : number of trees
        * learning_rate : learning rate
        * max_depth : tree max depth
        * min_samples_split : minimum number of samples to consider an internal node rule
        * min_samples_leaf : minimum number of samples to consider an leaf node rule
        * max_features : maximum number of features to consider at a node
    
    
    """
    models = GradientBoostingClassifier(n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features)
    models = models.fit(X, y) 
    dico_color={0:'blue',1:'white',2:'red'}
        # title for the plots
    titles = 'Gradient Boosted '+' '.join([str(n_estimators),str(learning_rate)])

        # Set-up 2x2 grid for plotting.
    fig, ax = plt.subplots(1, 1,figsize=(5,5))
        #plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    
    plot_contours(ax, models, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
   
    ax.set_title(titles)
       
    plt.show()
    
    xx = np.linspace(np.min(X0)-5, np.max(X0)+5, 100)
    yy = np.linspace(np.min(X1)-5, np.max(X1)+5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    
    y_pred = models.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # View probabilities:
    probas = models.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    
    plt.figure(figsize=(10*n_classes,10))
    for k in range(n_classes):
        plt.subplot(1, n_classes, k + 1)
        #plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel('Gradient Boosted')
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),extent=(np.min(X0)-5, np.max(X0)+5, np.min(X1)-5, np.max(X1)+5), origin='lower',cmap='plasma',alpha=0.7)
        plt.xticks(())
        plt.xlim([np.min(X0)-5, np.max(X0)+5])
        plt.ylim([np.min(X1)-5, np.max(X1)+5])
        plt.yticks(())
        plt.title('Class '+str(k),fontsize=25)
        
        idx = (y_pred == k)
        
        if idx.any():
            
            plt.scatter(X[idx, 0], X[idx, 1],s=100, marker='o', c=[dico_color[h] for h in y[idx]], edgecolor='k')

    ax = plt.axes([0,0,1,0.05])
    plt.title("Probability",fontsize=25)
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

    plt.show()