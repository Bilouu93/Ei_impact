import pandas as pd
from Importation import *
import numpy as np


def K_t(X):
    df = df_ts.copy()
    df['input'] = X
    return sum(df['input']*df['Coef reg kt non normalisé'])

w_t = K_t(np.array([100 for i in range(14)])) # worst case

#Modèle logarithmique, faible concavité
def kt_1(K):
    arg_1 = (np.exp(1)-1)/w_t
    arg_2 = np.exp(1)
    return np.log(arg_1*K(X)+arg_2)


#Modèle exponentielle décroissante, concavité plus forte
def kt_2(K):
    arg = -3*(K+w_t)/(w_t)
    return 1 - np.exp(arg)


alpha_ps = 0.004 #paramètre pour régler la référence sur p_s

# Fonction de régression de p_s

def K_s(X):
    df = df_ts.copy()
    df['input'] = X
    return alpha_ps*sum(df['input']*df['Coef reg ps non normalisé'])

w_s = K_s(np.array([100 for i in range(14)])) # worst case

#Regression logistique asymétrique
def p_s(K, beta=0.05):
    arg = np.exp(-K)
    return 1/(1+beta*arg)