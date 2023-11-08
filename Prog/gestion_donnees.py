# -*- coding: utf-8 -*-

#####
# VotreNom (VotreMatricule) .~= À MODIFIER =~.
###

import numpy as np


class GestionDonnees:
    def __init__(self, nb_train, nb_test, lineairement_sep):
        self.nb_train = nb_train
        self.nb_test = nb_test
        self.lineairement_sep = lineairement_sep

    def donnees_aleatoires(self, nb_data):
        """
        Fonction qui génère des données 2D aléatoires

        nb_data : nb de donnees générées
        """
        if self.lineairement_sep:
            nb_data_1_2 = int(nb_data / 2.0)
        else:
            nb_data_1_2 = int(nb_data / 3.0)
            nb_data_2 = nb_data - int(2.0 * nb_data / 3.0)

        x_1 = np.random.randn(nb_data_1_2, 2) + np.array([[5, 1]])  # Gaussienne centrée en mu_1_1=[5,1]
        t_1 = np.ones(nb_data_1_2)
        x_2 = np.random.randn(nb_data_1_2, 2) + np.array([[2, 3]])  # Gaussienne centrée en mu_2=[2,3]
        t_2 = np.zeros(nb_data_1_2)
        x = np.vstack([x_1, x_2])
        t = np.hstack([t_1, t_2])


        if not self.lineairement_sep:
            x_2 = np.random.randn(nb_data_2, 2) + np.array([[0, 4]])  # Gaussienne centrée en mu_1_1=[0,4]
            t_2 = np.ones(x_2.shape[0])

            # Fusionne toutes les données dans un seul ensemble
            x = np.vstack([x, x_2])
            t = np.hstack([t, t_2])

        # Mélange aléatoire des données
        p = np.random.permutation(len(t))
        x = x[p, :]
        t = t[p]

        return x, t

    def generer_donnees(self):
        """
        Fonction qui genere des donnees de test et d'entrainement.

        nb_train : nb de donnees d'entrainement
        nb_test : nb de donnees de test
        """
        x_train, t_train = self.donnees_aleatoires(self.nb_train)
        x_test, t_test = self.donnees_aleatoires(self.nb_test)

        return x_train, t_train, x_test, t_test

