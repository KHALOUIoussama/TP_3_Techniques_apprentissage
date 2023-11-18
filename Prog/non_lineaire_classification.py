# -*- coding: utf-8 -*-
#####
# Timothée Blanchy (timb1101)
# Khaloui Oussama (khao1201)
###

"""
Execution dans un terminal

Exemple:
   python non_lineaire_classification.py rbf 100 200 0 0

"""

import numpy as np
import sys
from map_noyau import MAPnoyau
import gestion_donnees as gd


def analyse_erreur(err_train, err_test, seuil_surapprentissage=30, seuil_sousapprentissage=30, seuil_bon_ajustement=5):
    """
    Fonction qui affiche un WARNING lorsqu'il y a apparence de sur ou de sous
    apprentissage
    """
    if err_test - err_train > seuil_surapprentissage:
        print('WARNING: Surapprentissage détecté. L\'erreur de test est considérablement plus élevée que l\'erreur d\'entraînement.')
    elif err_train > seuil_sousapprentissage and err_test > seuil_sousapprentissage:
        print('WARNING: Sous-apprentissage détecté. Les erreurs d\'entraînement et de test sont toutes deux élevées.')
    elif err_train < seuil_bon_ajustement and err_test < seuil_bon_ajustement:
        print('Bon ajustement: Les erreurs d\'entraînement et de test sont à des niveaux acceptables.')
    else:
        print('Situation incertaine: Il se peut que des ajustements supplémentaires soient nécessaires.')

def main():

    if len(sys.argv) < 6:
        usage = "\n Usage: python non_lineaire_classification.py type_noyau nb_train nb_test lin validation\
        \n\n\t type_noyau: rbf, lineaire, polynomial, sigmoidal\
        \n\t nb_train, nb_test: nb de donnees d'entrainement et de test\
        \n\t lin : 0: donnees non lineairement separables, 1: donnees lineairement separable\
        \n\t validation: 0: pas de validation croisee,  1: validation croisee\n"
        print(usage)
        return

    type_noyau = sys.argv[1]
    nb_train = int(sys.argv[2])
    nb_test = int(sys.argv[3])
    lin_sep = int(sys.argv[4])
    vc = bool(int(sys.argv[5]))

    # On génère les données d'entraînement et de test
    generateur_donnees = gd.GestionDonnees(nb_train, nb_test, lin_sep)
    [x_train, t_train, x_test, t_test] = generateur_donnees.generer_donnees()

    # On entraine le modèle
    mp = MAPnoyau(noyau=type_noyau)

    if vc is False:
        mp.entrainement(x_train, t_train)
    else:
        mp.validation_croisee(x_train, t_train)

    # Calcul de l'erreur sur l'ensemble d'entraînement
    err_train = 0
    N_train = x_train.shape[0]
    for i in range(N_train):
        err_train += mp.erreur(t_train[i], mp.prediction(x_train[i]))
    err_train = (err_train / N_train) * 100  # Erreur moyenne en pourcentage

    # Calcul de l'erreur sur l'ensemble de test
    err_test = 0
    N_test = x_test.shape[0]
    for i in range(N_test):
        err_test += mp.erreur(t_test[i], mp.prediction(x_test[i]))
    err_test = (err_test / N_test) * 100  # Erreur moyenne en pourcentage


    print('Erreur train = ', err_train, '%')
    print('Erreur test = ', err_test, '%')
    analyse_erreur(err_train, err_test)

    # Affichage
    mp.affichage(x_test, t_test)

if __name__ == "__main__":
    main()
