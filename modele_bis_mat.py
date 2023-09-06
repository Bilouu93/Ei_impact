import pandas as pd
import numpy as np
from Importation import dic_wb, df_wb, df_ts
from scipy.optimize import fsolve


delta = 0.015 #taux d'actualisation

alpha_ps = 0.004 #paramètre pour régler la référence sur p_s
climatic = 0.05
revenus = 0.06
nb_pas = 10
dic_pas = {'Années':1, 'Semestre':0.5, 'Trimestre':1/3, 'Mois':1/12, 'Jour':1/365}



#Worst case
df = df_ts.copy()
df['input'] = np.array([100 for i in range(14)])
wt = sum(df['input']*df['Coef reg kt non normalisé'])
ws = sum(df['input']*df['Coef reg ps non normalisé'])

#Fonction utile
def indice_max_inferieur(n, L):
    gauche, droite = 0, len(L) - 1
    indice = 0
    while gauche <= droite:
        milieu = (gauche + droite) // 2

        if L[milieu] < n:
            indice = milieu
            gauche = milieu + 1
        else:
            droite = milieu - 1

    return indice

def modele_transitoire(final, tau,t):
    return final*(1-np.exp(-t/(tau+0.001)))

def modele_cycle_de_vie(t, temps_de_croissance, valeur_maturite, temps_de_maturite):
    temps_declin_car = (temps_de_maturite - temps_de_croissance)/(2*5)
    if t < temps_de_croissance:
        return (valeur_maturite / temps_de_croissance) * t
    elif temps_de_croissance <= t < temps_de_maturite:
        return valeur_maturite
    elif temps_de_maturite <= t < temps_de_maturite + 5*temps_declin_car:
        return valeur_maturite * np.exp(-(t - temps_de_maturite) / (temps_declin_car))
    else:
        return 0.0

class Impact:
    def __init__(self,input_A,Trans,horizon, pas):
        self.input_A = input_A
        self.Trans = Trans
        self.horizon = horizon
        self.pas = pas
    
    def nb_gp(self):
        dic = self.input_A
        return len(list(dic.keys()))

    def N_pop(self):
        n = 1
        N = 0
        nb_gp = self.nb_gp()
        N_list = np.zeros(nb_gp+1)
        while n<nb_gp+1:
            N+=self.input_A[n][-1][2]
            N_list[n] = N
            n+=1
        return N_list


    def n_glob(self):
        N_list = self.N_pop()
        return N_list[-1]

    
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


    def f_WB(self, n, t):
        N_list = self.N_pop()
        gp = 1 + indice_max_inferieur(n,N_list)
        L_var = self.input_A[gp]
        n_tot = N_list[-1]
        tau_1 = L_var[-1][1]
        K = L_var[-1][2]
        tau_2 = L_var[-1][3]
        k = self.kt_2()
        p = self.ps()
        fac = (1/(1+delta))**(t*dic_pas[self.pas])
        def g(n, t): #Fonction de la taille du groupe liée à l'individu n, à l'instant t
            gp = 1 + indice_max_inferieur(n,N_list[1:])
            return modele_cycle_de_vie(t, tau_1, K, tau_2)
        def T(n): #Fonction: premier t tel l'individu n est touché
            rge_n = N_list[gp-1:gp+1]
            n_gp = rge_n[1] - rge_n[0]
            def equation_a_resoudre(t):
                return g(n,t) - n
            t1 = fsolve(equation_a_resoudre, x0=0)
            t2 = fsolve(equation_a_resoudre, x0=tau_2)
            return t1,t2
        t1,t2 = T(n)
        if t>t1 and t<t2:
            WB = 0
            for k in range(len(L_var)-1):
                delta_A = modele_cycle_de_vie(t-t1, L_var[k][1], L_var[k][2], L_var[k][3])
                if delta_A>0:
                    delta_X = k*delta_A
                else:
                    delta_X = (2-k)*delta_A
                WB += dic_wb[L_var[k][0]]*delta_X
            WB = p*fac*WB
        else:
            WB = 0
        return WB

      
    def fonction_impact_gp(self, n):
    # Calcule le WB sommé sur le temps pour un individu n
        T = np.linspace(0,self.horizon -1,self.horizon, dtype=int)
        print(f_WB(n, 2))
        return np.sum(np.array([self.f_WB(n, t) for t in T]))
    
    def fonction_impact_temps(self, t):
    # Calcule le WB sommé sur le temps pour un individu n
        n_tot = self.n_glob()
        N = np.linspace(1,int(n_tot),int(n_tot), dtype=int)
        return np.sum(np.array([self.f_WB(n, t) for n in N]))


    def fonction_impact_somme_gp(self):
        nb_gp = self.nb_gp()
        hist = np.zeros(nb_gp)
        N_pop = self.N_pop()
        cat = [str(i) for i in range(nb_gp)]
        for k in range(nb_gp):
            hist[k] = np.sum(np.array([self.fonction_impact_gp(n) for n in np.arange(N_pop[k],N_pop[k+1])]))
            cat[k] = f"groupe {k+1}"
        data = pd.DataFrame({'Groupes': cat,
                     'Impact': hist})
        return data


    def impact_tot(self):
    # Calcule l'impact brut
        T = np.linspace(0,self.horizon -1,self.horizon, dtype=int)
        return sum([self.fonction_impact_temps(t) for t in T])
        
'''    def __init__(self,input_A,Trans,horizon, argent, nec_clim, C02, pas):
        self.input_A = input_A
        self.Trans = Trans
        self.horizon = horizon
        self.argent = argent
        self.nec_clim = nec_clim
        self.CO2 = C02
        self.pas = pas'''