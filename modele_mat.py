import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Importation import dic_wb, df_wb, df_ts

delta = 0.015 #taux d'actualisation

alpha_ps = 0.004 #paramètre pour régler la référence sur p_s



#Worst case
df = df_ts.copy()
df['input'] = np.array([100 for i in range(14)])
wt = sum(df['input']*df['Coef reg kt non normalisé'])
ws = sum(df['input']*df['Coef reg ps non normalisé'])

#Fonction utile
def plus_proche_valeur(n,L):
    ind = 0
    if n>L[-1][1]:
        return len(L),0
    while n>L[ind][1]:
        ind+=1
    return ind, L[ind][0]

def modele_transitoire(final, tau,t):
    return final*(1-np.exp(-t/tau))

class Impact:
    def __init__(self,input_A,Trans,horizon):
        self.input_A = input_A
        self.Trans = Trans
        self.horizon = horizon

    def dic_A_n(self):
        dic_A_n = [] #Listes de dictionnaires des delta_A pour chaque instant
        for t in range(self.horizon):
            dic_t = {}
            for k in self.input_A.keys():
                dic_t[k] = [(self.input_A[k][ind][0], modele_transitoire(self.input_A[k][ind][1], self.input_A[k][ind][2], t)) for ind in range(len(self.input_A[k])) ]
            dic_A_n.append(dic_t)
        return dic_A_n
    
    def nb_gp(self):
        dic = self.input_A
        return len(list(dic.keys()))

    def n_glob(self):
        return sum([self.input_A[k][-1][1] for k in range(1,self.nb_gp()+1)])
    
    def Kt(self):
    # Fonction de régression de kt
        df = df_ts.copy()
        df['input'] = self.Trans
        return sum(df['input']*df['Coef reg kt non normalisé'])
    
    
    def kt_1(self):
    #Modèle logarithmique, faible concavité
        arg_1 = (np.exp(1)-1)/wt
        arg_2 = np.exp(1)
        return np.log(arg_1*Kt(self)+arg_2)

    def kt_2(self):
    #Modèle exponentielle décroissante, concavité plus forte
        arg = -3*(self.Kt()+wt)/(wt)
        return 1 - np.exp(arg)

    def Ks(self):
    #Fonction de régression de p_s
        df = df_ts.copy()
        df['input'] = self.Trans
        final = 0.004 * sum(df['input']*df['Coef reg ps non normalisé'])
        return final
    
    def ps(self, beta=0.05):
    #Regression logistique asymétrique pour la probabilité de succès
        arg = np.exp(-self.Ks())
        return 1/(1+beta*arg)


    def calcul_impact(self):
    #Calcul un dictionnaire qui associe (groupe,temps) à (valeur_wb, taille_tot_pop)
        k = self.kt_2()
        p = self.ps()
        dic_I = {}
        d_A_n = self.dic_A_n()
        for t in range(self.horizon):
            fac = (1/(1+delta))**(t/12)                 #facteur d'actualisation
            wb = 0
            n_tot = 0                                      #wb à l'instant t sommé sur tous les couples (groupe, variable)
            for gp, X in d_A_n[t].items():
                wb_gp = 0
                n = X[-1][1]
                for var_val in X[:-1]:
                    var = var_val[0]
                    A = var_val[1]
                    if A > 0:                               #Delta positifs ou négatifs
                        X = k * p * A  
                    else:
                        X = (2 - k) * p * A
                    wb_var = dic_wb[var] * X
                    wb_gp += wb_var
                n_tot += n
                dic_I[(gp,t)] = (wb_gp*fac, n_tot)
        return dic_I

    def fonction_WB_inf(self, n, t):
    # Calcule le WB d'un individu n à l'instant t
        dic_I = self.calcul_impact()
        sub_dic = {groupe: valeur for (groupe, temps), valeur in dic_I.items() if temps == t}
        L = list(sub_dic.values())
        return plus_proche_valeur(n,L)[1]
      
    def fonction_impact_gp(self, n):
    # Calcule le WB sommé sur le temps pour un individu n
        T = np.linspace(0,self.horizon -1,self.horizon, dtype=int)
        return sum([self.fonction_WB_inf(n, t) for t in T])
    
    def fonction_impact_temps(self, t):
    # Calcule le WB collectif à l'instant t
        d_A_n = self.dic_A_n()
        return sum([self.fonction_WB_inf(sum([d_A_n[t][k][-1][1]+0.1 for k in range(1,n)]) , t)*d_A_n[t][n][-1][1] for n in range(1,self.nb_gp()+1)])

    def impact_tot(self):
    # Calcule l'impact brut
        T = np.linspace(0,self.horizon -1,self.horizon, dtype=int)
        return sum([self.fonction_impact_temps(t) for t in T])

