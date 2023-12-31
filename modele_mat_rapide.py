import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Importation import *
from scipy.optimize import bisect


delta = 0.015 #taux d'actualisation

alpha_ps = 0.004 #paramètre pour régler la référence sur p_s
beta_xlim = 1.5 #Paramètre pour la prise en conmpte des ressources avec le modèle 1
alpha_per = 0.05 #Paramètre pour la variable transversale pérennité traitée à part
alpha_par = 0.75 #Paramètre pour la prise en conmpte des ressources avec le modèle 2

#Worst case pour les variables régressives de kt et ps
df = df_ts.copy()
df['input'] = np.array([100 for i in range(7)])
wt = sum(df['input']*df['Coef reg kt non normalisé'])
ws = sum(df['input']*df['Coef reg ps non normalisé'])

# Priors pour les coûts d'opportunité
climatic = 0.1
revenus = 0.4


class Impact:
    def __init__(self, input_A, Trans, horizon, pas, money_cost, env_cost, necessity, pas_indiv, ressources):
        self.input_A = input_A
        self.Trans = Trans
        self.horizon = horizon
        self.pas = pas
        self.money_cost = money_cost
        self.env_cost = env_cost
        self.necessity = necessity
        self.pas_indiv = pas_indiv
        self.ressources = ressources

        #On met ces termes dans l'initialisation pour ne pas avoir avoir à les recalculer
        self.nb_gp_value = self.nb_gp() 
        self.N_pop_tot = self.N_pop()
        self.n_glob_value = self.n_glob() 
        self.Kt_value = self.Kt()
        self.kt1_value = self.kt_1()
        self.kt2_value = self.kt_2()
        self.Ks_value = self.Ks()
        self.ps_value = self.ps()
        self.max_temps = self.max_temps_param()
        self.solutions = self.solutions_t1_t2()

    # Fonction auxiliaire pour la recherce d'éléemnts dans une liste
    def indice_max_inferieur(self, n, L):
        gauche, droite = 0, len(L) - 1
        indice = -1
        while gauche <= droite:
            milieu = (gauche + droite) // 2

            if L[milieu] < n:
                indice = milieu
                gauche = milieu + 1
            else:
                droite = milieu - 1
        return indice


    # Modèle du cycle de vie pour les variables dynamiques (spécifiques et taille de population)
    def cycle_de_vie(self, t, K_1, K_2, tau_glob, a, b):
        tau_1 = a * tau_glob
        tau_2 = b * tau_glob

        if t < 0:
            return 0
        elif 0 <= t < tau_glob:
            return (K_1 - K_2) * np.exp(-t / (tau_1 + 0.00001)) + K_2
        else:
            return ((K_1 - K_2) * np.exp(-tau_glob / (tau_1 + 0.000001)) + K_2) * np.exp(-(t - tau_glob) / (tau_2 + 0.000001))


    # Nombre de groupes
    def nb_gp(self):
        return len(list(self.input_A.keys()))


    # Liste contenant les valeurs des tailles de population en les sommant par groupe
    def N_pop(self):
        n = 1
        N = 0
        nb = self.nb_gp_value
        N_list = np.zeros(nb+1)
        N_list[0] = 1
        while n < nb+1:
            N += self.input_A[n][-1][1]
            N_list[n] = N
            n += 1
        return N_list

    # Nombre de personnes total
    def n_glob(self):
        N_list = self.N_pop_tot
        return N_list[-1]

    # Coût d'opportunité financier
    def money_cost_opp(self):
        return round(revenus*self.n_glob_value*self.pas_indiv*np.log((500 + self.money_cost/(100*self.n_glob_value))/500))
    
    # Cout d'opportunité environnemental
    def env_cost_opp(self):
        return round(climatic*self.n_glob_value*self.pas_indiv*self.necessity*40*(self.env_cost/(self.n_glob_value*self.pas_indiv*5))*(1/(1+delta))**(30))


    # Calcul de la variable régressive pour kt
    def Kt(self):
        df = df_ts.copy()
        df_2 = pd.DataFrame.from_dict(self.Trans, 'index', columns=['Valeur']).reset_index()
        df['input'] = df_2['Valeur']
        return sum(df['input']*df['Coef reg kt non normalisé'])


    # Calcul de kt avec faible concavité
    def kt_1(self):
        arg_1 = (np.exp(1)-1)/wt
        arg_2 = np.exp(1)
        kt = np.log(arg_1*self.Kt_value+arg_2)
        return kt

    # Calcul de kt avec plus grande concavité
    def kt_2(self):
        arg = -3*(self.Kt_value+wt)/(wt)
        return 1 - np.exp(arg)

    # Calcul de la variable régressive pour ps
    def Ks(self):
        df = df_ts.copy()
        df_2 = pd.DataFrame.from_dict(self.Trans, 'index', columns=['Valeur']).reset_index()
        df['input'] = df_2['Valeur']
        final = 0.004 * sum(df['input']*df['Coef reg ps non normalisé'])
        return final

    # Calcul de ps avec un modèle logistique discret
    def ps(self, beta=0.05):
        arg = np.exp(-self.Ks_value)
        p_s = 1/(1+beta*arg)
        return p_s

    # Donne le groupe d'une personne
    def groupe(self, n):
        N_list = self.N_pop_tot
        if n == 1:
            gp = 1
        else:
            gp = 1 + self.indice_max_inferieur(n, N_list)
        return gp

    # Le temps qu'il faut pour que le dernier individu ne soit plus concerné par une augmentation du WB
    def max_temps_param(self):
        nb_groupes = self.nb_gp_value
        L_temps = np.zeros(nb_groupes)
        for k in range(nb_groupes):
            tau_glob = self.input_A[k+1][-1][2]
            b = self.input_A[k+1][-1][4]
            L_temps[k] = tau_glob*(1+5*b)
        return max(L_temps)


    # Modèle 1 pour les ressources
    def transfo_1(self,x):
        return (1-np.exp(-x/beta_xlim))
    
    def delta_trans_1_var(self, var):
        R = self.ressources
        arg = np.sum(np.array([dic_EI[m][var]*R[m] for m in R.keys()]))
        if var != 'Pérennité':
            return ((100-(self.Trans[var]+100))*self.transfo_1(arg), arg)
        else:
            return arg*alpha_per

    def delta_trans_1(self):
        d_delta_var_1 = {var: self.delta_trans_1_var(var) for var in variables_EI}
        return d_delta_var_1

    # Modèle 2 pour les ressources
    def transfo_2(self, X):
        return np.log((100+X)/(100-X))

    def transfo_2_inv(self, y):
        def f(x):
            return  self.transfo_2(x) - y
        t_solution = bisect(f, 0, 100-1e-5, xtol=0.01, maxiter=50)
        return t_solution
        
    def delta_trans_2_var(self, var):
        R = self.ressources
        T = self.Trans
        arg_1 = np.sum(np.array([dic_EI[m][var]*R[m] for m in R.keys()]))
        if var != 'Pérennité':
            arg_avant = self.transfo_2(T[var]+100)
            arg_2 = (1+alpha_par*arg_1)*arg_avant
            return (self.transfo_2_inv(arg_2) - (T[var]+100), arg_avant, arg_2)
        else:
            return arg_1*alpha_per

    def delta_trans_2(self):
        d_delta_var_2 = {var: self.delta_trans_2_var(var) for var in variables_EI}
        return d_delta_var_2


    # Instants auquel un individu n souscrit au service et le quitte
    def solution(self, n):
        gp = self.groupe(n)
        L_var = self.input_A[gp]
        tau_pop_glob = L_var[-1][2]
        K_2_n = L_var[-1][1]
        a_n = L_var[-1][3]
        b_n = L_var[-1][4]
        FIFO = L_var[-1][5]
        K_1_n = 0
        N_popul = self.N_pop_tot
        if gp != 1:
            n_app = n - N_popul[gp-1]
        else:
            n_app = n

        def g_1(t):
            return self.cycle_de_vie(t,  K_1_n, K_2_n, tau_pop_glob, a_n, b_n) - n_app

        def g_2(t):
            return self.cycle_de_vie(t, K_1_n, K_2_n, tau_pop_glob, a_n, b_n) - (K_2_n-n_app)

        try:
            t_solution_1 = bisect(g_1, 0, tau_pop_glob, xtol=0.01, maxiter=50)
            bool_1 = True
        except (ValueError, RuntimeError):
            t_solution_1 = self.horizon
            bool_1 = False
        if FIFO:
            try:
                t_solution_2 = bisect(g_2, tau_pop_glob, tau_pop_glob * (1 + 5*b_n), xtol=0.01, maxiter=50)
                bool_2 = True
            except (ValueError, RuntimeError):
                t_solution_2 = False
                bool_2 = False
        else:
            try:
                t_solution_2 = bisect(g_1, tau_pop_glob, tau_pop_glob * (1 + 5*b_n), xtol=0.01, maxiter=50)
                bool_2 = True
            except (ValueError, RuntimeError):
                t_solution_2 = False
                bool_2 = False   

        return [t_solution_1, t_solution_2], [bool_1,bool_2]


    # Instants auquel un individu n souscrit au service et le quitte pour chaque individu
    def solutions_t1_t2(self):
        N_tot = self.n_glob_value
        D_solution = {}
        for k in range(1, int(N_tot) + 1):
            sol = self.solution(k)
            D_solution[k] = sol
        return D_solution


    # Fonction micro du WB
    def f_WB(self, n, t):
        Solution = self.solutions
        gp = self.groupe(n)
        kt = self.kt2_value
        p = self.ps_value
        fac = (1/(1+delta))**(t*self.pas)
        L_var = self.input_A[gp]
        t_solution_1, t_solution_2 = Solution[n][0][0], Solution[n][0][1]
        bool_1, bool_2 = Solution[n][1][0], Solution[n][1][1]
        condition_1 = t_solution_1 < t < t_solution_2 and bool_1 is True and bool_2 is True
        condition_2 = t_solution_1 < t and bool_2 is False
        if condition_1 or condition_2:
            WB = 0
            for k in range(len(L_var) - 1):
                tau_par = L_var[k][3]
                K_1 = L_var[k][1]
                K_2 = L_var[k][2]
                a = L_var[k][4]
                b = L_var[k][5]
                delta_A = self.cycle_de_vie(t - t_solution_1, K_1, K_2, tau_par, a, b)
                if delta_A > 0:
                    delta_X = kt * delta_A
                else:
                    delta_X = (2 - kt) * delta_A
                WB += dic_wb[L_var[k][0]] * delta_X
            WB = p * fac * WB
        else:
            WB = 0
        return WB*self.pas_indiv


    # WB de l'individu n sommé sur le temps
    def fonction_impact_gp(self, n):
        temps_int = self.max_temps
        N_fen = int(temps_int)
        T = np.linspace(0, temps_int, N_fen)
        L_tot = np.array([self.f_WB(n,t) for t in T])
        S = np.sum(L_tot)
        return S * self.pas

    # WB collectif de l'ensemble de la population à un instant t
    def fonction_impact_temps(self, t):
        N_tot = self.n_glob_value
        N_indiv = np.arange(1, N_tot+1)
        return np.sum(np.array([self.f_WB(n,t) for n in N_indiv]))

    # WB collectif d'un groupe de la population à un instant t
    def fonction_impact_temps_gp(self, gp, t):
        N_popul = self.N_pop_tot
        nb_groupes = self.nb_gp_value
        N_indiv_gp = np.arange(int(N_popul[gp-1]), int(N_popul[gp])+1)
        return np.sum(np.array([self.f_WB(n,t) for n in N_indiv_gp]))

    #Tableau donnant les WB sommé sur le temps pour chaque groupe
    def fonction_impact_somme_gp(self):
        nb_groupes = self.nb_gp_value
        hist = np.zeros(nb_groupes)
        N_popul = self.N_pop_tot
        cat = [str(i) for i in range(nb_groupes)]
        for k in range(nb_groupes):
            hist[k] = np.sum(np.array([self.fonction_impact_gp(n) for n in np.arange(int(N_popul[k]), int(N_popul[k+1]))]))
            cat[k] = f"Groupe {k+1}"
        data = pd.DataFrame({'Groupes': cat, 'Impact': hist})
        return data


    # Impact brut total
    def impact_tot(self):
        N_tot = self.n_glob_value
        N_indiv = np.arange(1, N_tot+1)
        L_indiv = np.array([self.fonction_impact_gp(n) for n in N_indiv])
        return np.sum(L_indiv)