

# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)



# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X,y,NumSpks,title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    #ax = plt.subplot(111)
    #colors = cm.rainbow(np.linspace(0, 1, 20))
    for i in range(X.shape[0]):
        #plt.scatter(X[i, 0], X[i, 1], color=colors[y[i]]) 
        plt.scatter(X[i, 0], X[i, 1], color=plt.cm.Set1(y[i] / NumSpks)) 
        """
    for i in range(X.shape[0]):
        #plt.scatter(X[i, 0], X[i, 1], color=plt.cm.Set1(y[i] / NumSpks)) 
        plt.scatter(X[i, 0], X[i, 1], color=plt.cm.Set1(y[i]  / NumSpks)) 
        """
    
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)    
# ----------------------------------------------------------------------



    
# ----------------------------------------------------------------------
def TsneSpk(embedings,labels,NumSpks):
    # ----------------------
    #randomly select spks!!!!!
    X=embedings #RANDOM EMBEDING
    y=labels     #THEIR LABELS
    n_samples, n_features = X.shape
    #save selected speakers to compare methods!!!!!
    # ---------------------
    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne,y,NumSpks,title='Tsne plot')
# ----------------------------------------------------------------------
