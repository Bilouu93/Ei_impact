import pandas as pd
import numpy as np
from Importation import dic_wb, df_wb, df_ts
from scipy.optimize import fsolve


delta = 0.015 #taux d'actualisation

alpha_ps = 0.004 #paramètre pour régler la référence sur p_s
climatic = 0.1
revenus = 0.4

input_A = {1: [('No_Criminality/Insecurity', 0.0, 0.28589818676332773, 30.0, 0.2, 0.3), ('No_Discrimination', 0.0, 0.2386595301527497, 30.0, 0.2, 0.3), ('Social coherence (cohésion sociale)', 0.0, 0.24562876497194663, 30.0, 0.2, 0.3), ('Traditionalism', 0.0, 0.2462321338011842, 30.0, 0.2, 0.3), ('Years in state', 0.0, 0.4928416967230111, 30.0, 0.2, 0.3), ('Academic education', 0.0, 0.4814990545027433, 30.0, 0.2, 0.3), ('Family education', 0.0, 0.49183568418079376, 30.0, 0.2, 0.3), ('Access to local services', 0.0, 0.22043003907445524, 30.0, 0.2, 0.3), ('Air and water pollution', -0.0, -0.146190884230072, 30.0, 0.2, 0.3), ('Close to green spaces', 0.0, 0.4098599629309733, 30.0, 0.2, 0.3), ('Housing quality', 0.0, 0.39386305593491194, 30.0, 0.2, 0.3), ('No_No accès to health services', 0.0, 0.4542328886907622, 30.0, 0.2, 0.3), ('No_No access to food', 0.0, 0.5979117290980132, 30.0, 0.2, 0.3), ('No_No access to water', 0.0, 0.2772791418811836, 30.0, 0.2, 0.3), ('Noise pollution', 0.0, 0.17077771951701087, 30.0, 0.2, 0.3), ('Richessness fauna, flaura and landscape', -0.0, -0.04446696486803142, 30.0, 0.2, 0.3), ('Relative health', 0.0, 0.5082015244961546, 30.0, 0.2, 0.3), ('Relative salary', 0.0, 0.05205277273146558, 30.0, 0.2, 0.3), ('Family support', 0.0, 0.10482757050642288, 30.0, 0.2, 0.3), ('No_Famlily strain', 0.0, 0.2879776223637999, 30.0, 0.2, 0.3), ('Friends support', 0.0, 0.540825462457915, 30.0, 0.2, 0.3), ('Having grandparents', 0.0, 0.296630055054001, 30.0, 0.2, 0.3), ('Number of children', 0.0, 0.1261885847576002, 30.0, 0.2, 0.3), ('Partenaired', 0.0, 0.5703947198380546, 30.0, 0.2, 0.3), ('Positive relationships', -0.0, -0.07497894306906769, 30.0, 0.2, 0.3), ('Pregnancy', 0.0, 0.2543751723800801, 30.0, 0.2, 0.3), ('Social Closeness', 0.0, 0.22200077664357204, 30.0, 0.2, 0.3), ('Social integration', -0.0, -0.15373866902233965, 30.0, 0.2, 0.3), ('No_Tobacco and alcohol of others', 0.0, 0.38795752736889483, 30.0, 0.2, 0.3), ('No_Contribution to others', 0.0, 0.17673901293902555, 30.0, 0.2, 0.3), ('Emotional management', 0.0, 0.5854136889980963, 30.0, 0.2, 0.3), ('Managing impact of past experience', 0.0, 0.12042928135234343, 30.0, 0.2, 0.3), ('Positive habits', 0.0, 0.40584864732331977, 30.0, 0.2, 0.3), ('Stress management', 0.0, 0.3810276258882475, 30.0, 0.2, 0.3), ('Coping active (proactivity)', 0.0, 0.33021928440717957, 30.0, 0.2, 0.3), ('Efforts in life', -0.0, -0.06137869071774693, 30.0, 0.2, 0.3), ('No_Hypervigilence (=somatic amplification)', 0.0, 0.47430390760100755, 30.0, 0.2, 0.3), ('Selective secondary control(=commitment)', 0.0, 0.4171247949870343, 30.0, 0.2, 0.3), ('Compensatory primary control', 0.0, 0.4372681943002498, 30.0, 0.2, 0.3), ('Openness to new experiences', 0.0, 0.1133763788893849, 30.0, 0.2, 0.3), ('Evenement mastery', 0.0, 0.5116649927142307, 30.0, 0.2, 0.3), ('Goals in life', -0.0, -0.19584334237860804, 30.0, 0.2, 0.3), ('Self -control', 0.0, 0.20493302482584452, 30.0, 0.2, 0.3), ('Time management and anticipation', 0.0, 0.2103731688427506, 30.0, 0.2, 0.3), ('(Need of Security)', 0.0, 0.5304156377293621, 30.0, 0.2, 0.3), ('Altruism', -0.0, -0.07571644142580417, 30.0, 0.2, 0.3), ('Gratitude', -0.0, -0.05555923096156956, 30.0, 0.2, 0.3), ('Honesty', 0.0, 0.3660873715527125, 30.0, 0.2, 0.3), ('No_Agreebleness', -0.0, -0.029402239469416447, 30.0, 0.2, 0.3), ('Extraversion', 0.0, 0.57338460168794, 30.0, 0.2, 0.3), ('No_Health locus of control', 0.0, 0.43566920445127805, 30.0, 0.2, 0.3), ('Optimism', 0.0, 0.38816398050692297, 30.0, 0.2, 0.3), ('No_Social potency', -0.0, -0.03526962712243656, 30.0, 0.2, 0.3), ('No_Sympathy', 0.0, 0.16477463151553579, 30.0, 0.2, 0.3), ('Consciensciousness', 0.0, 0.1795380813792452, 30.0, 0.2, 0.3), ('Living the moment', 0.0, 0.4576607322974355, 30.0, 0.2, 0.3), ('Physical attractiveness', 0.0, 0.5142635602758863, 30.0, 0.2, 0.3), ('Self protection', 0.0, 0.23261171606099057, 30.0, 0.2, 0.3), ('Self-efficacy', 0.0, 0.5064281622873661, 30.0, 0.2, 0.3), ('No_Alcohol problems', 0.0, 0.35687892811952265, 30.0, 0.2, 0.3), ('No_Drug consumption', -0.0, -0.08528850430228135, 30.0, 0.2, 0.3), ('Food', 0.0, 0.3024241390755776, 30.0, 0.2, 0.3), ('Social networks', 0.0, 0.4252826865928305, 30.0, 0.2, 0.3), ('No_Tobacco', 0.0, 0.25047931489961767, 30.0, 0.2, 0.3), ('No_Mental health professsionnal', 0.0, 0.16352359733175764, 30.0, 0.2, 0.3), ('No_Aches', -0.0, -0.05690090670171097, 30.0, 0.2, 0.3), ('Diseases', 0.0, 0.14228662205084525, 30.0, 0.2, 0.3), ('No_Sleep problems', 0.0, 0.41729973495117595, 30.0, 0.2, 0.3), ('Games', -0.0, -0.11791528967097503, 30.0, 0.2, 0.3), ('Intellectual activities', 0.0, 0.08776310650350005, 30.0, 0.2, 0.3), ('Physical activity', 0.0, 0.2812155312173367, 30.0, 0.2, 0.3), ('Political participation', 0.0, 0.46434615747432856, 30.0, 0.2, 0.3), ('Sex life', 0.0, 0.5164110248878038, 30.0, 0.2, 0.3), ('Spiritual and religious experiences', 0.0, 0.41035477069937415, 30.0, 0.2, 0.3), ('Assessment financial situation', 0.0, 0.17490946737710822, 30.0, 0.2, 0.3), ('Assessment professionnal life', 0.0, 0.28449704712092183, 30.0, 0.2, 0.3), ('Balance btw personal and professional life', 0.0, 0.20193032139821504, 30.0, 0.2, 0.3), ('Control over professionnal life', 0.0, 0.02770812900219638, 30.0, 0.2, 0.3), ('Financial situation effort', 0.0, 0.4231325667658224, 30.0, 0.2, 0.3), ('taille', 19, 30.0, 0.3, 0.3, True)], 2: [('No_Criminality/Insecurity', -0.0, -0.2093324300060515, 30.0, 0.2, 0.3), ('No_Discrimination', 0.0, 0.08298842590353805, 30.0, 0.2, 0.3), ('Social coherence (cohésion sociale)', -0.0, -0.3897412824443828, 30.0, 0.2, 0.3), ('Traditionalism', -0.0, -0.38529656400295015, 30.0, 0.2, 0.3), ('Years in state', -0.0, -0.4416371405523007, 30.0, 0.2, 0.3), ('Academic education', 0.0, 0.0795727043238823, 30.0, 0.2, 0.3), ('Family education', -0.0, -0.04558782346686591, 30.0, 0.2, 0.3), ('Access to local services', -0.0, -0.34693597205521853, 30.0, 0.2, 0.3), ('Air and water pollution', 0.0, 0.052990789827878526, 30.0, 0.2, 0.3), ('Close to green spaces', -0.0, -0.2145414028631431, 30.0, 0.2, 0.3), ('Housing quality', -0.0, -0.3656914799426731, 30.0, 0.2, 0.3), ('No_No accès to health services', -0.0, -0.38324877378489036, 30.0, 0.2, 0.3), ('No_No access to food', -0.0, -0.13489722977596164, 30.0, 0.2, 0.3), ('No_No access to water', -0.0, -0.44557315936345626, 30.0, 0.2, 0.3), ('Noise pollution', -0.0, -0.39595360054507617, 30.0, 0.2, 0.3), ('Richessness fauna, flaura and landscape', -0.0, -0.12096362965267787, 30.0, 0.2, 0.3), ('Relative health', -0.0, -0.10993519395639179, 30.0, 0.2, 0.3), ('Relative salary', -0.0, -0.35094911889646374, 30.0, 0.2, 0.3), ('Family support', 0.0, 0.02831386042913986, 30.0, 0.2, 0.3), ('No_Famlily strain', -0.0, -0.2469549109501148, 30.0, 0.2, 0.3), ('Friends support', -0.0, -0.021839035186676403, 30.0, 0.2, 0.3), ('Having grandparents', -0.0, -0.38341973888911884, 30.0, 0.2, 0.3), ('Number of children', -0.0, -0.3685089493809255, 30.0, 0.2, 0.3), ('Partenaired', -0.0, -0.25881786953748054, 30.0, 0.2, 0.3), ('Positive relationships', -0.0, -0.34428465664069885, 30.0, 0.2, 0.3), ('Pregnancy', -0.0, -0.2984379142468499, 30.0, 0.2, 0.3), ('Social Closeness', -0.0, -0.1209121477970474, 30.0, 0.2, 0.3), ('Social integration', 0.0, 0.03771897721108819, 30.0, 0.2, 0.3), ('No_Tobacco and alcohol of others', -0.0, -0.18787440546067175, 30.0, 0.2, 0.3), ('No_Contribution to others', 0.0, 0.09204892930490838, 30.0, 0.2, 0.3), ('Emotional management', -0.0, -0.2926139893896076, 30.0, 0.2, 0.3), ('Managing impact of past experience', -0.0, -0.17840105709949272, 30.0, 0.2, 0.3), ('Positive habits', -0.0, -0.26630687206391057, 30.0, 0.2, 0.3), ('Stress management', -0.0, -0.15085053815411714, 30.0, 0.2, 0.3), ('Coping active (proactivity)', -0.0, -0.037698810620697076, 30.0, 0.2, 0.3), ('Efforts in life', -0.0, -0.08226884536749157, 30.0, 0.2, 0.3), ('No_Hypervigilence (=somatic amplification)', -0.0, -0.42331491681145117, 30.0, 0.2, 0.3), ('Selective secondary control(=commitment)', -0.0, -0.3931105231470405, 30.0, 0.2, 0.3), ('Compensatory primary control', -0.0, -0.24578997039582612, 30.0, 0.2, 0.3), ('Openness to new experiences', 0.0, 0.03411363200040829, 30.0, 0.2, 0.3), ('Evenement mastery', 0.0, 0.07824862919776909, 30.0, 0.2, 0.3), ('Goals in life', 0.0, 0.038162126911333805, 30.0, 0.2, 0.3), ('Self -control', 0.0, 0.05459384560382918, 30.0, 0.2, 0.3), ('Time management and anticipation', -0.0, -0.23713628709716494, 30.0, 0.2, 0.3), ('(Need of Security)', -0.0, -0.34619422268893435, 30.0, 0.2, 0.3), ('Altruism', -0.0, -0.2231516302819901, 30.0, 0.2, 0.3), ('Gratitude', -0.0, -0.399212480716471, 30.0, 0.2, 0.3), ('Honesty', -0.0, -0.3554560086159213, 30.0, 0.2, 0.3), ('No_Agreebleness', -0.0, -0.4016293292622746, 30.0, 0.2, 0.3), ('Extraversion', -0.0, -0.05550434659471254, 30.0, 0.2, 0.3), ('No_Health locus of control', -0.0, -0.48601440859001743, 30.0, 0.2, 0.3), ('Optimism', -0.0, -0.2578850309011733, 30.0, 0.2, 0.3), ('No_Social potency', -0.0, -0.2433162286959687, 30.0, 0.2, 0.3), ('No_Sympathy', -0.0, -0.3031006313515232, 30.0, 0.2, 0.3), ('Consciensciousness', -0.0, -0.12566823403863275, 30.0, 0.2, 0.3), ('Living the moment', -0.0, -0.19107210858963636, 30.0, 0.2, 0.3), ('Physical attractiveness', 0.0, 0.07304851943832813, 30.0, 0.2, 0.3), ('Self protection', 0.0, 0.015675985774256107, 30.0, 0.2, 0.3), ('Self-efficacy', -0.0, -0.2861093598865986, 30.0, 0.2, 0.3), ('No_Alcohol problems', -0.0, -0.3003144306352937, 30.0, 0.2, 0.3), ('No_Drug consumption', -0.0, -0.127093635996467, 30.0, 0.2, 0.3), ('Food', -0.0, -0.09847361357762585, 30.0, 0.2, 0.3), ('Social networks', 0.0, 0.03933623735530123, 30.0, 0.2, 0.3), ('No_Tobacco', -0.0, -0.33550627443001835, 30.0, 0.2, 0.3), ('No_Mental health professsionnal', -0.0, -0.2031369553906039, 30.0, 0.2, 0.3), ('No_Aches', -0.0, -0.023166098168841964, 30.0, 0.2, 0.3), ('Diseases', -0.0, -0.4274735077167539, 30.0, 0.2, 0.3), ('No_Sleep problems', 0.0, 0.026236802556900307, 30.0, 0.2, 0.3), ('Games', 0.0, 0.057253176747861456, 30.0, 0.2, 0.3), ('Intellectual activities', 0.0, 0.02949942003483008, 30.0, 0.2, 0.3), ('Physical activity', -0.0, -0.10066054120902834, 30.0, 0.2, 0.3), ('Political participation', -0.0, -0.02759797168909167, 30.0, 0.2, 0.3), ('Sex life', 0.0, 0.07485921846353027, 30.0, 0.2, 0.3), ('Spiritual and religious experiences', -0.0, -0.30722876528923215, 30.0, 0.2, 0.3), ('Assessment financial situation', -0.0, -0.3680319051821237, 30.0, 0.2, 0.3), ('Assessment professionnal life', -0.0, -0.1527822454861374, 30.0, 0.2, 0.3), ('Balance btw personal and professional life', -0.0, -0.06317450312776729, 30.0, 0.2, 0.3), ('Control over professionnal life', -0.0, -0.10106855818435528, 30.0, 0.2, 0.3), ('Financial situation effort', -0.0, -0.3636799493066575, 30.0, 0.2, 0.3), ('taille', 11, 30.0, 0.3, 0.3, True)]}

Trans = [-71., -71., -71., -50., -50., -11., -50., -51., -51., -27., -49., -60., -43., -43.]
horizon = 40


#Worst case
df = df_ts.copy()
df['input'] = np.array([100 for i in range(14)])
wt = sum(df['input']*df['Coef reg kt non normalisé'])
ws = sum(df['input']*df['Coef reg ps non normalisé'])

import numpy as np
import pandas as pd
from scipy.optimize import bisect

class Impact:
    def __init__(self, input_A, Trans, horizon, pas, money_cost, env_cost, necessity, pas_indiv):
        self.input_A = input_A
        self.Trans = Trans
        self.horizon = horizon
        self.pas = pas
        self.money_cost = money_cost
        self.env_cost = env_cost
        self.necessity = necessity
        self.pas_indiv = pas_indiv

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

    def cycle_de_vie(self, t, K_1, K_2, tau_glob, a, b):
        tau_1 = a * tau_glob
        tau_2 = b * tau_glob

        if t < 0:
            return 0
        elif 0 <= t < tau_glob:
            return (K_1 - K_2) * np.exp(-t / (tau_1 + 0.00001)) + K_2
        else:
            return ((K_1 - K_2) * np.exp(-tau_glob / (tau_1 + 0.000001)) + K_2) * np.exp(-(t - tau_glob) / (tau_2 + 0.000001))



    def nb_gp(self):
        return len(list(self.input_A.keys()))

    def N_pop(self):
        n = 1
        N = 0
        nb = self.nb_gp()
        N_list = np.zeros(nb+1)
        N_list[0] = 1
        while n < nb+1:
            N += self.input_A[n][-1][1]
            N_list[n] = N
            n += 1
        return N_list

    def n_glob(self):
        N_list = self.N_pop()
        return N_list[-1]

    def money_cost_opp(self):
        return round(revenus*self.n_glob()*self.pas_indiv*np.log((500 + self.money_cost/(100*self.n_glob()))/500))
    
    def env_cost_opp(self):
        print((climatic, self.n_glob(), self.necessity,self.env_cost, delta))
        return round(climatic*self.n_glob()*self.pas_indiv*self.necessity*40*(self.env_cost/(self.n_glob()*self.pas_indiv*5))*(1/(1+delta))**(30))

    def Kt(self):
        df = df_ts.copy()
        df['input'] = self.Trans
        return sum(df['input']*df['Coef reg kt non normalisé'])

    def kt_1(self):
        arg_1 = (np.exp(1)-1)/wt
        arg_2 = np.exp(1)
        return np.log(arg_1*self.Kt()+arg_2)

    def kt_2(self):
        arg = -3*(self.Kt()+wt)/(wt)
        return 1 - np.exp(arg)

    def Ks(self):
        df = df_ts.copy()
        df['input'] = self.Trans
        final = 0.004 * sum(df['input']*df['Coef reg ps non normalisé'])
        return final

    def ps(self, beta=0.05):
        arg = np.exp(-self.Ks())
        return 1/(1+beta*arg)

    def groupe(self, n):
        N_list = self.N_pop()
        if n == 1:
            gp = 1
        else:
            gp = 1 + self.indice_max_inferieur(n, N_list)
        return gp

    def max_temps_param(self):
        nb_groupes = self.nb_gp()
        L_temps = np.zeros(nb_groupes)
        for k in range(nb_groupes):
            tau_glob = self.input_A[k+1][-1][2]
            b = self.input_A[k+1][-1][4]
            L_temps[k] = tau_glob*(1+5*b)
        return max(L_temps)

    def solution(self, n):
        gp = self.groupe(n)
        L_var = self.input_A[gp]
        tau_pop_glob = L_var[-1][2]
        K_2_n = L_var[-1][1]
        a_n = L_var[-1][3]
        b_n = L_var[-1][4]
        FIFO = L_var[-1][5]
        K_1_n = 0
        N_popul = self.N_pop()
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

    def solutions_t1_t2(self):
        N_tot = self.n_glob()
        D_solution = {}
        for k in range(1, int(N_tot) + 1):
            sol = self.solution(k)
            D_solution[k] = sol
        return D_solution

    def f_WB(self, n, t, Solution):
        gp = self.groupe(n)
        kt = self.kt_2()
        p = self.ps()
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

    def fonction_impact_gp(self, n, Solution):
        temps_int = self.max_temps_param()
        N_fen = int(temps_int)
        T = np.linspace(0, temps_int, N_fen)
        L_tot = np.array([self.f_WB(n,t,Solution) for t in T])
        S = np.sum(L_tot)
        return S * self.pas

    def fonction_impact_temps(self, t, Solution):
        N_tot = self.n_glob()
        N_indiv = np.arange(1, N_tot+1)
        return np.sum(np.array([self.f_WB(n,t,Solution) for n in N_indiv]))

    def fonction_impact_temps_gp(self, gp, t, Solution):
        N_popul = self.N_pop()
        nb_groupes = self.nb_gp()
        N_indiv_gp = np.arange(int(N_popul[gp-1]), int(N_popul[gp])+1)
        return np.sum(np.array([self.f_WB(n,t,Solution) for n in N_indiv_gp]))


    def fonction_impact_somme_gp(self, Solution):
        nb_groupes = self.nb_gp()
        hist = np.zeros(nb_groupes)
        N_popul = self.N_pop()
        cat = [str(i) for i in range(nb_groupes)]
        for k in range(nb_groupes):
            hist[k] = np.sum(np.array([self.fonction_impact_gp(n, Solution) for n in np.arange(int(N_popul[k]), int(N_popul[k+1]))]))
            cat[k] = f"Groupe {k+1}"
        data = pd.DataFrame({'Groupes': cat, 'Impact': hist})
        return data

    def impact_tot(self, Solution):
        N_tot = self.n_glob()
        N_indiv = np.arange(1, N_tot+1)
        L_indiv = np.array([self.fonction_impact_gp(n,Solution) for n in N_indiv])
        return np.sum(L_indiv)

# pas = 2
# # Exemple d'utilisation
# if __name__ == "__main__":
#     # Créez une instance de la classe Impact en passant les arguments nécessaires
#     impact_calculator = Impact(input_A, Trans, horizon, pas)

#     # Utilisez les méthodes de la classe Impact pour effectuer les calculs
#     solutions = impact_calculator.solutions_t1_t2()
#     impact_gp = impact_calculator.fonction_impact_somme_gp(solutions)
#     impact_tot = impact_calculator.impact_tot(solutions)
#     f_WB = impact_calculator.f_WB
#     imp_temp = impact_calculator.fonction_impact_temps_gp
#     im_tp = impact_calculator.fonction_impact_temps

#     # Affichez les résultats
#     print("Solutions t1 et t2 pour chaque individu:", solutions)
#     #print("f_WB:", imp_temp(1, 0.5, solutions) + imp_temp(2, 0.5, solutions))
#     print("Impact total:", impact_tot)

