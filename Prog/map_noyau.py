# -*- coding: utf-8 -*-

#####
# Timothée Blanchy (timb1101)
# Khaloui Oussama (khao1201)
###

import numpy as np
import matplotlib.pyplot as plt


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
        N = len(x_train)

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
    
    # def validation_croisee(self, x_tab, t_tab):
    #     """
    #     Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
    #     ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
    #     ``self.lamb`` avec une validation croisée de type "k-fold" où k=10 avec les
    #     données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
    #     trouvés, le modèle est entraîné une dernière fois.

    #     SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
    #     de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
    #     de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
    #     """
    #     # Validation croisée (k=10)
    #     k = 10
    #     N = len(x_tab)
    #     best_erreur = np.inf
    #     if N < k:
    #         raise ValueError('Nombre de données trop petit')
        
    #     indices = np.random.permutation(N)

    #     # Division des données en k parties
    #     indices = np.array_split(indices, k)

    #     # Recherche des meilleurs paramètres
    #     nb_recherche = 1000
    #     for i in range(nb_recherche):
    #         self.sigma_square = np.random.uniform(0.000000001, 2)
    #         self.lamb = np.random.uniform(0.000000001, 2)
    #         self.c = np.random.uniform(0, 5)
    #         self.b = np.random.uniform(0.00001, 0.01)
    #         self.d = np.random.uniform(0.00001, 0.01)
    #         self.M = np.random.randint(2, 7)

    #         for i in range(k):
    #             # Création des données d'entrainement et de validation
    #             x_train = np.delete(x_tab, indices[i], axis=0)
    #             t_train = np.delete(t_tab, indices[i], axis=0)
    #             x_val = x_tab[indices[i]]
    #             t_val = t_tab[indices[i]]

    #             # Entrainement
    #             self.entrainement(x_train, t_train)

    #             # Calcul de l'erreur
    #             erreur = 0
    #             for j in range(len(x_val)):
    #                 erreur += self.erreur(t_val[j], self.prediction(x_val[j]))
    #             erreur /= N

    #             # Mise à jour des meilleurs paramètres
    #             if erreur < best_erreur:
    #                 best_erreur = erreur
    #                 best_sigma_square = self.sigma_square
    #                 best_lamb = self.lamb
    #                 best_c = self.c
    #                 best_b = self.b
    #                 best_d = self.d
    #                 best_M = self.M

    #     # Affichage des meilleurs paramètres
    #     print('Meilleurs paramètres:')
    #     print(f"sigma_square: {best_sigma_square}")
    #     print(f"lamb: {best_lamb}")
    #     print(f"c: {best_c}")
    #     print(f"b: {best_b}")
    #     print(f"d: {best_d}")
    #     print(f"M: {best_M}")

    #     # Mise à jour des paramètres
    #     self.sigma_square = best_sigma_square
    #     self.lamb = best_lamb
    #     self.c = best_c
    #     self.b = best_b
    #     self.d = best_d
    



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
        N = len(x_tab)
        best_erreur = np.inf
        if N < k:
            raise ValueError('Nombre de données trop petit')
        
        indices = np.random.permutation(N)

        # Division des données en k parties
        indices = np.array_split(indices, k)
        nb_tests = 10

        if self.noyau == 'rbf':   # Paramètres à tester : sigma_square, lamb
            for sigma_square in np.linspace(0.000000001, 2, nb_tests):
                for lamb in np.linspace(0.000000001, 2, nb_tests):
                    erreur = 0
                    for i in range(k):
                        # Création des données d'entrainement et de validation
                        x_train = np.delete(x_tab, indices[i], axis=0)
                        t_train = np.delete(t_tab, indices[i], axis=0)
                        x_val = x_tab[indices[i]]
                        t_val = t_tab[indices[i]]

                        # Entrainement
                        self.sigma_square = sigma_square
                        self.lamb = lamb
                        self.entrainement(x_train, t_train)

                        # Calcul de l'erreur
                        for j in range(len(x_val)):
                            erreur += self.erreur(t_val[j], self.prediction(x_val[j]))
                    erreur /= N

                    # Mise à jour des meilleurs paramètres
                    # print(f"sigma_square: {sigma_square} lamb: {lamb} erreur: {erreur}")
                    if erreur < best_erreur:
                        best_erreur = erreur
                        best_sigma_square = sigma_square
                        best_lamb = lamb
                        # print(f'Meilleurs paramètres: sigma_square: {best_sigma_square} lamb: {best_lamb}')
        
            print(f'Meilleurs paramètres:\nsigma_square: {best_sigma_square}\nlamb: {best_lamb}')
            self.sigma_square = best_sigma_square
            self.lamb = best_lamb

        elif self.noyau == 'lineaire':   # Paramètres à tester : lamb
            for lamb in np.linspace(0.000000001, 2, nb_tests):
                erreur = 0
                for i in range(k):
                    # Création des données d'entrainement et de validation
                    x_train = np.delete(x_tab, indices[i], axis=0)
                    t_train = np.delete(t_tab, indices[i], axis=0)
                    x_val = x_tab[indices[i]]
                    t_val = t_tab[indices[i]]

                    # Entrainement
                    self.lamb = lamb
                    self.entrainement(x_train, t_train)

                    # Calcul de l'erreur
                    for j in range(len(x_val)):
                        erreur += self.erreur(t_val[j], self.prediction(x_val[j]))
                erreur /= N

                # Mise à jour des meilleurs paramètres
                # print(f"lamb: {lamb} erreur: {erreur}")
                if erreur < best_erreur:
                    best_erreur = erreur
                    best_lamb = lamb
                    # print(f'Meilleurs paramètres: lamb: {best_lamb} best_erreur: {best_erreur}')

            print(f'Meilleurs paramètres:\nlamb: {best_lamb}')
            self.lamb = best_lamb

        elif self.noyau == 'polynomial':   # Paramètres à tester : lamb, c, M
            for lamb in np.linspace(0.000000001, 2, nb_tests):
                for c in np.linspace(0, 5, 10):
                    for M in range(2, 7):
                        erreur = 0
                        for i in range(k):
                            # Création des données d'entrainement et de validation
                            x_train = np.delete(x_tab, indices[i], axis=0)
                            t_train = np.delete(t_tab, indices[i], axis=0)
                            x_val = x_tab[indices[i]]
                            t_val = t_tab[indices[i]]

                            # Entrainement
                            self.lamb = lamb
                            self.c = c
                            self.M = M
                            self.entrainement(x_train, t_train)

                            # Calcul de l'erreur
                            for j in range(len(x_val)):
                                erreur += self.erreur(t_val[j], self.prediction(x_val[j]))
                        erreur /= N

                        # Mise à jour des meilleurs paramètres
                        # print(f"lamb: {lamb} c: {c} M: {M} erreur: {erreur}")
                        if erreur < best_erreur:
                            best_erreur = erreur
                            best_lamb = lamb
                            best_c = c
                            best_M = M
                            # print(f'Meilleurs paramètres: lamb: {best_lamb} c: {best_c} M: {best_M} erreur: {best_erreur}')
            
            print(f'Meilleurs paramètres:\nlamb: {best_lamb}\nc: {best_c}\nM: {best_M}')
            self.lamb = best_lamb
            self.c = best_c
            self.M = best_M

        elif self.noyau == 'sigmoidal':   # Paramètres à tester : lamb, b, d
            for lamb in np.linspace(0.000000001, 2, nb_tests):
                for b in np.linspace(0.00001, 0.01, nb_tests):
                    for d in np.linspace(0.00001, 0.01, nb_tests):
                        erreur = 0
                        for i in range(k):
                            # Création des données d'entrainement et de validation
                            x_train = np.delete(x_tab, indices[i], axis=0)
                            t_train = np.delete(t_tab, indices[i], axis=0)
                            x_val = x_tab[indices[i]]
                            t_val = t_tab[indices[i]]

                            # Entrainement
                            self.lamb = lamb
                            self.b = b
                            self.d = d
                            self.entrainement(x_train, t_train)

                            # Calcul de l'erreur
                            for j in range(len(x_val)):
                                erreur += self.erreur(t_val[j], self.prediction(x_val[j]))
                        erreur /= N

                        # Mise à jour des meilleurs paramètres
                        # print(f"lamb: {lamb} b: {b} d: {d} erreur: {erreur}")
                        if erreur < best_erreur:
                            best_erreur = erreur
                            best_lamb = lamb
                            best_b = b
                            best_d = d
                            # print(f'Meilleurs paramètres: lamb: {best_lamb} b: {best_b} d: {best_d} erreur: {best_erreur}')
            
            print(f'Meilleurs paramètres:\nlamb: {best_lamb}\nb: {best_b}\nd: {best_d}')
            self.lamb = best_lamb
            self.b = best_b
            self.d = best_d

        else:
            raise ValueError('Noyau invalide')



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