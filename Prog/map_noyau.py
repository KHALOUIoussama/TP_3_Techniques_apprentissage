# -*- coding: utf-8 -*-

#####
# Timothée Blanchy (timb1101)
# Khaloui Oussama (khao1201)
###

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, olynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None

        

    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        self.x_train = x_train
        N = x_train.shape[0]

        if self.noyau == 'rbf':
            sq_norm = (x_train ** 2).sum(axis=1)
            k = (-2*np.dot(x_train, x_train.T) + sq_norm.reshape(-1, 1) + sq_norm)/(-2*self.sigma_square)
            k = np.exp(k)
        else:
            k = self.noyau_fonction(x_train, x_train)

        self.a = np.dot(np.linalg.inv(self.lamb * np.identity(N) + k), t_train)


        
    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        k_x = self.noyau_fonction(x, self.x_train)
        y = np.dot(k_x.T, self.a)
        return 1 if y > 0.5 else 0
    

    def noyau_fonction(self, x, x_train):
        """
        Retourne la valeur du noyau désiré (rbf, lineaire, polynomial ou sigmoidal)
        pour une entrée ``x`` et les données d'entrainement ``x_train``.
        """
        if self.noyau == 'rbf':
            sq_norm = (x ** 2).sum()
            k = (-2*np.dot(x_train, x.T) + np.dot(x_train, x) + sq_norm)/(-2*self.sigma_square)
            return np.exp(k)
        elif self.noyau == 'lineaire':
            return np.dot(x_train, x.T)
        elif self.noyau == 'polynomial':
            return np.power(np.dot(x_train, x.T) + self.c, self.M)
        elif self.noyau == 'sigmoidal':
            return np.tanh(np.dot(x_train, x.T)*self.b + self.d)     
        else:
            raise ValueError('Noyau invalide')

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        return (t - prediction) ** 2    



    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=10 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        # Validation croisée (k=10)
        k = 10
        N = x_tab.shape[0]
        if N < k:
            raise ValueError('Nombre de données trop petit')

        # Créer les plis pour la validation croisée
        indices = np.arange(N)
        np.random.shuffle(indices)
        indices = np.array_split(indices, k)

        # Initialisation des meilleurs paramètres et de l'erreur
        best_erreur = np.inf
        best_params = {}

        # Définition des grilles de recherche pour chaque hyperparamètre
        sigma_squares = np.linspace(0.000000001, 2, 20)
        lambs = np.linspace(0.000000001, 2, 20)
        cs = np.linspace(0, 5, 20)
        bs = np.linspace(0.00001, 0.01, 20)
        ds = np.linspace(0.00001, 0.01, 20)
        Ms = range(2, 7)

        # Sélection des hyperparamètres en fonction du noyau choisi
        if self.noyau == 'rbf':
            parametres_grid = {'sigma_square': sigma_squares, 'lamb': lambs}
        elif self.noyau == 'lineaire':
            parametres_grid = {'lamb': lambs}
        elif self.noyau == 'polynomial':
            parametres_grid = {'lamb': lambs, 'c': cs, 'M': Ms}
        elif self.noyau == 'sigmoidal':
            parametres_grid = {'lamb': lambs, 'b': bs, 'd': ds}
        else:
            raise ValueError('Noyau invalide')

        # Recherche en grille avec validation croisée
        for params_combination in tqdm(itertools.product(*parametres_grid.values()), desc='Grid Search'):
            params = dict(zip(parametres_grid.keys(), params_combination))
            erreur_moyenne = 0
            
            for i, val_index in enumerate(indices):
                train_index = np.hstack([indices[j] for j in range(k) if j != i])
                x_train, x_val = x_tab[train_index], x_tab[val_index]
                t_train, t_val = t_tab[train_index], t_tab[val_index]
                
                # Mise à jour des paramètres du modèle
                for param, value in params.items():
                    setattr(self, param, value)
                self.entrainement(x_train, t_train)
                
                # Calcul de l'erreur sur le pli de validation
                predictions = np.array([self.prediction(x) for x in x_val])
                erreur_moyenne += np.mean((predictions - t_val) ** 2)
            
            erreur_moyenne /= k
            if erreur_moyenne < best_erreur:
                best_erreur = erreur_moyenne
                best_params = params.copy()

        # Affichage des meilleurs paramètres et mise à jour du modèle
        print(f'Meilleurs paramètres:\n{best_params}')
        for param, value in best_params.items():
            setattr(self, param, value)
        self.entrainement(x_tab, t_tab)




    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()