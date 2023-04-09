
#-------------------------------------------------------------------
# -- LIMITES AFFICHAGE DE TOUS DATAFRAMES --
#-----------------------------------------------------------------------
dico_pd_option = {
    'display.max_rows': 400,    # nbre max de lignes 
    'display.max_column': 200,  # nbre max de colonnes
    'display.width': 100,       # largeur lignes en px je pense
    'display.precision': 5,     # precision des valeurs
    'display.max_colwidth': 200  # largeurs colonnes en px
}
for cle, val in dico_pd_option.items():
    pd.set_option(cle, val)   # on limite pour tous dataframe pandas les affichages 
    
    
#-------------------------------------------------------------------
# -- VOIR FRONTIERES DE DECISION POUR CLUSTERING, CLASSIFICATION OU REGRESSION --
#-----------------------------------------------------------------------
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
    
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=.5)
    
    plt.scatter(X[:,0], X[:,1], c=y, alpha=0.8, edgecolors='k')

""" plot_decision_boundary(model, X_train, y_train) pour l'utiliser, clf = model dans def """

#-------------------------------------------------------------------
# --  --
#-----------------------------------------------------------------------