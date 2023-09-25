import streamlit as st
import numpy as np
#from modele_mat import *
#from modele_bis_mat import *
from modele_mat_rapide import *
import random
import plotly.express as px
import plotly.graph_objects as go
from fonction_parametrage import *
import pickle
from Importation import *
import plotly.colors as pc
import locale
import copy



L_var = list(dic_wb.keys())

L_trans = variables_EI[:len(variables_EI)-1]




# Définissez la configuration régionale pour formater les nombres avec des espaces pour les milliers
locale.setlocale(locale.LC_ALL, '')

# Fonction pour formater un nombre avec des espaces pour les milliers
def format_number(number):
    return locale.format_string("%d", number, grouping=True)


def main():

#-------------------------------------------------------------------------------------------------------------------------------------
# Mise en forme
#-------------------------------------------------------------------------------------------------------------------------------------

    st.set_page_config(layout="wide")
    st.image("EI_logo.png", width=200)
    st.markdown("""
    <style>
        .centered-title {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # Afficher le titre centré
    st.markdown('<h1 class="centered-title">Modèle de prédiction d\'impact</h1>', unsafe_allow_html=True)

    style = """ 
        .fixed-author {
            position: fixed;
            bottom: 0;
            left: 0;
            margin: 10px;
            font-style: italic;
        }
    """

    st.markdown(f'<style>{style}</style>', unsafe_allow_html=True)

    st.markdown('<p class="fixed-author">Bilel HATMI</p>', unsafe_allow_html=True)

#-------------------------------------------------------------------------------------------------------------------------------------
#Choix de la dynamique
#-------------------------------------------------------------------------------------------------------------------------------------

    st.header("1- Paramètres globaux")

    EI = st.checkbox("Déterminer la contibution d'Elements Impact dans le projet")

    unit_temps = ['Années','Semestre', 'Trimestre', 'Mois', 'Jour']
    unit_indiv = ['Millions','Centaine de milliers', 'Dizaine de milliers', 'Milliers', 'Centaine', 'Dizaine', 'Unitaire']
    dic_pas_temps = {'Années':1, 'Semestre':0.5, 'Trimestre':1/3, 'Mois':1/12, 'Jour':1/365}
    dic_pas_indiv = {unit_indiv[0]: 1e6, unit_indiv[1]: 1e5, unit_indiv[2]: 1e4, unit_indiv[3]: 1e3, unit_indiv[4]: 1e2, unit_indiv[5]: 1e1, unit_indiv[6]: 1}

    default_selection_temps = unit_temps.index('Semestre')
    default_selection_indiv = unit_indiv.index('Centaine')

    col_1, col_2 = st.columns(2)
    with col_1:
        unit_temps = st.radio("Choisir une unité temporelle:", unit_temps, index = default_selection_temps)
    with col_2:
        unit_indiv = st.radio("Choisir une unité de comptage des individus:", unit_indiv, index = default_selection_indiv)

    pas_temps = dic_pas_temps[unit_temps]
    pas_indiv = dic_pas_indiv[unit_indiv]

    







    # bool_indiv = True
    # if option_tau == 'Macro':
    #     tau_global = st.number_input(r"Transitoire  $ \ \tau_{glob}$:", value = 1.0, step=0.1) 
    #     bool_indiv = False
    horizon = st.number_input(f"Taille de la fenêtre  $ \ T$:", value = 50, step=1)

#-------------------------------------------------------------------------------------------------------------------------------------
#Choix des variables transversales
#-------------------------------------------------------------------------------------------------------------------------------------


    st.header("2- Caractérisation des variables transversales")
    Trans = {}
    with st.expander("Cliquez ici pour modifier la valeur des variables transversales"):
        col1,col2,col3 = st.columns([5,1,5])
        with col1:
            for n in range(0,int((len(L_trans)/2)+1)):
                t_rd = random.randint(0,100)
                trans_value = st.slider(f"Valeur _{L_trans[n]}_:",
                                        value=50, 
                                        min_value=0, max_value=100, step=1)
                Trans[L_trans[n]] = trans_value -100
        with col3:
            for n in range(int((len(L_trans)/2)+1), len(L_trans)):
                t_rd = random.randint(0,100)
                trans_value = st.slider(f"Valeur _{L_trans[n]}_:",
                                        value=50, 
                                        min_value=0, max_value=100, step=1)
                Trans[L_trans[n]] = trans_value-100

#-------------------------------------------------------------------------------------------------------------------------------------
#Cout du projet
#-------------------------------------------------------------------------------------------------------------------------------------
    st.header("3- Les coûts d'opportunité")
    money_cost = st.number_input("Coût financier du projet en millier d'euros", value=1000, step=1)
    money_cost *= 1000
    env_cost = st.number_input('Coût environnemental du projet en tonnes de CO2eq', value=1000, step=1)
    necessity = st.number_input('Nécessité dans le contexte environnemental', value=1.00, step=0.01)

#-------------------------------------------------------------------------------------------------------------------------------------
#Gropues de population
#-------------------------------------------------------------------------------------------------------------------------------------

    st.header("4- Caractérisation des groupes de population")
    # Demander à l'utilisateur de saisir le nombre de groupes
    nb_gp = st.number_input("Nombre de groupes:", value = 2, min_value=1, step=1)

    A_input = {}
    N_pop = []

    options_tau = ['Macro','Micro']
    option_tau = st.radio("Caractérisez le régime transitoire des variables spécifiques:", options_tau)
    # Saisie des variables pour chaque groupe
    for group_id in range(1, nb_gp + 1):
        n_rd = random.uniform(100,200)
        A_input[group_id] = [[str(i), 0.0, 0.0, 0.0, 0.0] for i in range(len(L_var) + 1)]


        st.write(f"## <b>Groupe {group_id}</b>", unsafe_allow_html=True)
        with st.expander("Cliquez ici pour modifier la valeur des variables prédictives"):
            st.write(f"### <b>Tailles de population</b>", unsafe_allow_html=True)
            taille_pop = st.number_input(f"$N_{group_id}$ Taille du groupe:", 
                                            value = int(n_rd), 
                                            step=10, key=f"group{group_id}_taille" )

            tau_pop_glob = st.number_input(r"$\tau_{n_{glob}}$ Croissance:", 
                                        value = 30.00, 
                                        step=0.01, key=f"group{group_id}_pop_tau_1" )

            a_pop = st.number_input(r"$a_{n_{glob}}$ Croissance:", 
                                        value = 0.3, 
                                        step=0.01, key=f"group{group_id}_a_pop" )

            b_pop = st.number_input(r"$b_{n_{glob}}$ Déclin:", 
                            value = 0.3, 
                            step=0.01, key=f"group{group_id}_b_pop" )


            ordres = ['FIFO','FILO']
            ordre = st.radio("Choisir un mode de sélection pour caractériser le régime transitoire:", ordres, key=f"group{group_id}_ordre")
            if ordre == 'FIFO':
                FIFO = True
            else:
                FIFO = False


            st.markdown("---")
            A_input[group_id][-1] = ['taille', taille_pop, tau_pop_glob, a_pop, b_pop, FIFO]




            if option_tau == 'Macro':
                st.write(f"### <b>Dynamique macro des variables prédictives</b>", unsafe_allow_html=True)
                fraction_K1_glob = st.number_input(f"$K_1$ Phase d'adaptation (fraction RP):", 
                                                value = 0, 
                                                step=10, key=f"group{group_id}_K1" )

                tau_glob = st.number_input(r"$\tau_{glob}$ Croissance:", 
                                            value = 30.00, 
                                            step=0.01, key=f"group{group_id}_tau_glob" )

                a_glob = st.number_input(r"$a_{glob}$ Croissance:", 
                                            value = 0.2, 
                                            step=0.01, key=f"group{group_id}_a_glob" )

                b_glob = st.number_input(r"$b_{glob}$ Déclin:", 
                                value = 0.3, 
                                step=0.01, key=f"group{group_id}_b_glob" )

                st.markdown("---")
            st.write(f"### <b>Remplissage des valeurs</b>", unsafe_allow_html=True)
            col1,col2,col3 = st.columns([5,1,5])
            with col1:
                for n in range(int(len(L_var)/2)+1):
                    if group_id==1:
                        rd_value = random.uniform(-0.2,0.6)
                    elif group_id==2:
                        rd_value = random.uniform(-0.5,0.1)
                    else:
                        rd_value = random.uniform(-0.2,1)
                    variable_value = st.number_input(f"Valeur _{L_var[n]}_:", 
                                                        value = rd_value, 
                                                        step=0.01, key=f"group{group_id}_{L_var[n]}" )


                    if option_tau == 'Micro':
                        fraction_K1 = st.number_input(f"$K_1$ Phase d'adaptation (fraction RP):", 
                                                        value = 0, 
                                                        step=10, key=f"group{group_id}_K1_{L_var[n]}" )

                        tau = st.number_input(r"$\tau$ Croissance:", 
                                                    value = 1.00, 
                                                    step=0.01, key=f"group{group_id}_tau_{L_var[n]}" )

                        a = st.number_input(r"$a$  Croissance:", 
                                                    value = 0.1, 
                                                    step=0.01, key=f"group{group_id}_a_glob_{L_var[n]}" )

                        b = st.number_input(r"$b$  Déclin:", 
                                        value = 0.1, 
                                        step=0.01, key=f"group{group_id}_b_glob_{L_var[n]}" )                        
                    else:
                        fraction_K1 = fraction_K1_glob
                        tau = tau_glob
                        a = a_glob
                        b = b_glob
                    A_input[group_id][n] = [L_var[n], fraction_K1*variable_value, variable_value, tau, a, b]
                    st.markdown("---")
            with col3:
                for n in range(int(len(L_var)/2) + 1, len(L_var)):
                    if group_id==1:
                        rd_value = random.uniform(-0.2,0.6)
                    elif group_id==2:
                        rd_value = random.uniform(-0.5,0.1)
                    else:
                        rd_value = random.uniform(-0.2,1)
                    variable_value = st.number_input(f"Valeur _{L_var[n]}_:", 
                                                        value = rd_value, 
                                                        step=0.01, key=f"group{group_id}_{L_var[n]}" )


                    if option_tau == 'Micro':
                        fraction_K1 = st.number_input(f"$K_1$ Phase d'adaptation (fraction RP):", 
                                                        value = 0, 
                                                        step=10, key=f"group{group_id}_K1_{L_var[n]}" )

                        tau = st.number_input(r"$\tau$  Croissance:", 
                                                    value = 1.00, 
                                                    step=0.01, key=f"group{group_id}_tau_{L_var[n]}" )

                        a = st.number_input(r"$a$  Croissance:", 
                                                    value = 0.1, 
                                                    step=0.01, key=f"group{group_id}_a_glob_{L_var[n]}" )

                        b = st.number_input(r"$b$  Déclin:", 
                                        value = 0.1, 
                                        step=0.01, key=f"group{group_id}_b_glob_{L_var[n]}" )                        
                    else:
                        fraction_K1 = fraction_K1_glob
                        tau = tau_glob
                        a = a_glob
                        b = b_glob
                    A_input[group_id][n] = [L_var[n], fraction_K1*variable_value, variable_value, tau, a, b]
                    st.markdown("---")
                
    
    #st.write(str(A_input))
    #A_input = B_input

    #A_input = C_grp_input
    #Trans = Trans_AB
    compteur_titres = 4
    compteur_titres +=1
    if not EI:
        impact_class = Impact(A_input,Trans,horizon, pas_temps, money_cost, env_cost, necessity, pas_indiv, None)
        impact_value = round(impact_class.impact_tot())
        n_tot = impact_class.n_glob()
        money = impact_class.money_cost_opp()
        envt = impact_class.env_cost_opp()
        cat = ['Impact brut', 'Impact contrefactuel net', "Coût d'opportunité financier", "Coût d'opportunité environnemental"]
        colors = {cat[0]: "blue", cat[1]: "green", cat[2]: "red", cat[3]: "magenta"}
        list_impact = [impact_value, impact_value - money - envt, money, envt]
        df_imp = pd.DataFrame({'Type Impacts/Coûts': cat, 'Valeur': list_impact})

    

    if EI:
        st.header(f"{compteur_titres}- L'allocation des ressources")
        compteur_titres +=1
        data_EI = []
        for m in metier_EI:
            for i in range(1,6):
                data_EI.append([m, i, 4, 1])

        df_EI = pd.DataFrame(data_EI, columns=['Métier', 'Expertise', 'Ressources staff', 'Ressources EI'])
        edited_EI = st.data_editor(df_EI)
        df_EI_copy = df_EI.copy()
        df_EI_copy['num'] = (df_EI_copy['Ressources EI']*df_EI_copy['Expertise'])
        df_EI_copy['den'] = (df_EI_copy['Ressources EI']+df_EI_copy['Ressources staff'])*df_EI_copy['Expertise']
        df_ressources = (df_EI_copy.groupby('Métier')['num'].sum() / df_EI_copy.groupby('Métier')['den'].sum()).reset_index()

        df_ressources = df_ressources.rename(columns={0: 'augmentation_ressources'}).set_index('Métier')['augmentation_ressources']
        ressources = df_ressources.to_dict()
        impact_class = Impact(A_input,Trans,horizon, pas_temps, money_cost, env_cost, necessity, pas_indiv, ressources)
        impact_value = round(impact_class.impact_tot())
        n_tot = impact_class.n_glob()
        money = impact_class.money_cost_opp()
        envt = impact_class.env_cost_opp()

        



        delta_trans_1 = impact_class.delta_trans_1()
        delta_trans_2 = impact_class.delta_trans_2()
        transfo_1 = impact_class.transfo_1
        transfo_2_inv = impact_class.transfo_2_inv


        
        tuples_1 = list(delta_trans_1.values())
        tuples_2 = list(delta_trans_2.values())

        categories = list(delta_trans_1.keys())[:-1]
        trans = (np.array([list(Trans.values())])+100)[0]
        values_1_w = np.array([t[0] for t in tuples_1[:-1]])
        values_2_w = np.array([t[0] for t in tuples_2[:-1]])
        values_1 = trans + values_1_w
        values_2 = trans + values_2_w

        Trans_EI = dict(zip(categories, values_1-100))
        B_input = copy.deepcopy(A_input)
        k_per = 1 + tuples_1[-1]
        for i in range(1,nb_gp+1):
            B_input[i][-1][3] *= 1/(k_per)
            B_input[i][-1][4] *= 1/(k_per)
            B_input[i][-1][2] *= k_per 
        for i in range(1,nb_gp+1):
            for k in range(len(B_input[1])-1):
                B_input[i][k][4] *= 1/(k_per)
                B_input[i][k][5] *= 1/(k_per)
                B_input[i][k][3] *= k_per

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.write(B_input)
        # with col2:
        #     st.write(A_input)

        impact_class_EI = Impact(B_input,Trans_EI,horizon, pas_temps, money_cost, env_cost, necessity, pas_indiv, ressources)
        #Solutions = impact_class.solutions_t1_t2()
        impact_EI = round(impact_class_EI.impact_tot())
        money_EI = impact_class_EI.money_cost_opp()
        envt_EI = impact_class_EI.env_cost_opp()
        

        cat = ['Impact contrefactuel net', "Coût d'opportunité financier", "Coût d'opportunité environnemental"]*2
        list_impact = [impact_value - money - envt, money, envt, impact_EI - money_EI - envt_EI, money_EI, envt_EI]
        intervention_EI = ['Sans EI']*3 + ['Avec EI']*3
        df_imp = pd.DataFrame({'Type Impacts/Coûts': cat, 'Valeur': list_impact, 'EI': intervention_EI})
        

#-------------------------------------------------------------------------------------------------------------------------------------
#Vérification paramétrage:
#-------------------------------------------------------------------------------------------------------------------------------------


    st.header(f'{compteur_titres}- Vérification du paramétrage de $k_t$ et $p_s$')
    compteur_titres +=1
    with st.expander("Cliquez ici pour vérifier le parametrage des fonction $k_t$ et $p_s$"):
        Kt_list = np.linspace(-w_t,0,100)
        Ks_list = np.linspace(-w_s,0,100)
        KT = [kt_2(K) for K in Kt_list]
        PS = [p_s(K) for K in Ks_list]
        kt_true = impact_class.kt_2()
        ps_true = impact_class.ps()
        Kt_true = impact_class.Kt()
        Ks_true = impact_class.Ks()
        if EI:
            kt_true_EI = impact_class_EI.kt_2()
            ps_true_EI = impact_class_EI.ps()
            Kt_true_EI = impact_class_EI.Kt()
            Ks_true_EI = impact_class_EI.Ks()


        col1, col2 = st.columns(2)
        black_with_opacity = 'rgba(0, 0, 0, 0.4)'

        with col1:
            # Créer un graphe interactif Plotly
            fig_4 = px.line(x=Kt_list, y=KT, title='Fonction de pénalisation contrefactuelle')
            fig_4.update_traces(line=dict(color='orange', dash='solid'))
            fig_4.update_xaxes(title_text='K_kt')
            fig_4.update_yaxes(title_text='k_t')
            fig_4.update_layout(
                xaxis=dict(showgrid=True, gridcolor=black_with_opacity),
                yaxis=dict(showgrid=True, gridcolor=black_with_opacity), width =650
            )
            highlight_trace_4 = go.Scatter(x=[Kt_true], y=[kt_true], mode='markers', marker=dict(color='blue', size=10), name ='Actuel')
            fig_4.add_trace(highlight_trace_4)

            if EI:
                highlight_trace_5 = go.Scatter(x=[Kt_true_EI], y=[kt_true_EI], mode='markers', marker=dict(color='green', size=10), name ='EI')
                fig_4.add_trace(highlight_trace_5)
            st.plotly_chart(fig_4)
            if EI:
                st.write(f'Efficacité conctrefactuelle:    {round(kt_true_EI/kt_true,2)}')


        with col2:
            # Créer un graphe interactif Plotly
            fig_5 = px.line(x=Ks_list, y=PS, title='Fonction de probabilité de succès')
            fig_5.update_traces(line=dict(color='orange', dash='solid'))
            fig_5.update_xaxes(title_text='K_ps')
            fig_5.update_yaxes(title_text='p_s')
            fig_5.update_layout(
                xaxis=dict(showgrid=True, gridcolor=black_with_opacity),
                yaxis=dict(showgrid=True, gridcolor=black_with_opacity), width=650
            )
            #highlight_trace_5 = px.scatter(x=[Ks_true], y=[ps_true] )
            highlight_trace_6 = go.Scatter(x=[Ks_true], y=[ps_true], mode='markers', marker=dict(color='blue', size=10), name='Actuel')
            fig_5.add_trace(highlight_trace_6)

            if EI:
                highlight_trace_7 = go.Scatter(x=[Ks_true_EI], y=[ps_true_EI], mode='markers', marker=dict(color='green', size=10), name='EI')
                fig_5.add_trace(highlight_trace_7)
            st.plotly_chart(fig_5)
            if EI:
                st.write(f'Efficacité de succès:    {round(ps_true_EI/ps_true,2)}')



    if EI:
        st.header(str(compteur_titres)+'- Vérification du paramétrage du layer EI')
        compteur_titres +=1
        with st.expander("Cliquez ici pour vérifier le parametrage pour le layer EI"):

            # st.write(str(ressources))
            # st.write(str(Trans))
            

            L_tabs = st.tabs(variables_EI[:-1])
            for i in range(len(L_tabs)):
                with L_tabs[i]:
                    X_1 = np.linspace(0,10)
                    Y_1 = np.array([(100-(Trans[variables_EI[i]]+100))*transfo_1(x) for x in X_1]) + Trans[variables_EI[i]]+100
                    X_2 = np.linspace(0,7)
                    Y_2 = [transfo_2_inv(x) for x in X_2]
                    arg_1 = tuples_1[i][1]
                    arg_av = tuples_2[i][1]
                    arg_ap = tuples_2[i][2]
            
                    fig_var_1 = go.Figure()
                    fig_var_1.add_trace(go.Scatter(x=X_1, y=Y_1, mode='lines', name='Modèle 1'))
                    fig_var_1.add_hrect(y0=Trans[variables_EI[i]]+100, 
                    y1= (100-(Trans[variables_EI[i]]+100))*transfo_1(arg_1) +Trans[variables_EI[i]]+100, fillcolor= 'green', opacity=0.4, line_width=0)

                    fig_var_1.update_xaxes(title_text=f'Variable intermédiaire')
                    fig_var_1.update_yaxes(title_text="Variable transversale")
                    fig_var_1.update_layout(width = 650, title = 'Modèle 1')
                    fig_var_1.add_annotation(
                            go.layout.Annotation(
                                x=6,  # x-coordinate of the arrowhead
                                y=(100-(Trans[variables_EI[i]]+100))*transfo_1(arg_1) +Trans[variables_EI[i]]+100,  # y-coordinate of the arrowhead
                                xref="x",  # x-coordinate reference ("x" axis)
                                yref="y",  # y-coordinate reference ("y" axis)
                                ax=6,  # x-coordinate of the arrow tail
                                ay=Trans[variables_EI[i]]+96,  # y-coordinate of the arrow tail
                                axref="x",  # x-coordinate reference for the arrow tail ("x" axis)
                                ayref="y",  # y-coordinate reference for the arrow tail ("y" axis)
                                showarrow=True,
                                arrowhead=2,  # Arrowhead style (2 is a filled arrowhead)
                                #arrowsize=1.5,  # Arrowhead size
                                arrowwidth=2,  # Arrow width
                                text = 'Delta_X',
                                font = dict(size=16)
                            ),
                        )
                    fig_var_1.add_annotation(x=0, y=Trans[variables_EI[i]]+100,
                    text="X avant",
                    showarrow=True,
                    arrowhead=1)
                    fig_var_1.add_annotation(x=arg_1, y=(100-(Trans[variables_EI[i]]+100))*transfo_1(arg_1) +Trans[variables_EI[i]]+100,
                    text="X_EI",
                    showarrow=True,
                    arrowhead=1)
                    


                    fig_var_1.update_layout(
                        yaxis=dict(range=[0,100]) 
                    )

                    fig_var_2 = go.Figure()
                    fig_var_2.add_trace(go.Scatter(x=X_2, y=Y_2, mode='lines', name='modele_2'))
                    fig_var_2.add_hrect(y0=transfo_2_inv(arg_av), 
                    y1= transfo_2_inv(arg_ap), fillcolor= 'green', opacity=0.4, line_width=0)
                    fig_var_2.update_xaxes(title_text=f'Variable intermédiaire')
                    fig_var_2.update_yaxes(title_text="Variable transversale")
                    fig_var_2.update_layout(width = 650, title = 'Modèle 2')
                    fig_var_2.add_annotation(x=arg_av, y=transfo_2_inv(arg_av),
                    text="X_EI",
                    showarrow=True,
                    arrowhead=1)
                    fig_var_2.add_annotation(x=arg_ap, y=transfo_2_inv(arg_ap),
                    text="X avant",
                    showarrow=True,
                    arrowhead=1)
                    fig_var_2.add_annotation(
                            go.layout.Annotation(
                                x=5,  # x-coordinate of the arrowhead
                                y=transfo_2_inv(arg_ap),  # y-coordinate of the arrowhead
                                xref="x",  # x-coordinate reference ("x" axis)
                                yref="y",  # y-coordinate reference ("y" axis)
                                ax=5,  # x-coordinate of the arrow tail
                                ay=transfo_2_inv(arg_av)-4,  # y-coordinate of the arrow tail
                                axref="x",  # x-coordinate reference for the arrow tail ("x" axis)
                                ayref="y",  # y-coordinate reference for the arrow tail ("y" axis)
                                showarrow=True,
                                arrowhead=2,  # Arrowhead style (2 is a filled arrowhead)
                                #arrowsize=1.5,  # Arrowhead size
                                arrowwidth=2,  # Arrow width
                                text = 'Delta_X',
                                font = dict(size=16)
                            ),
                        )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_var_1)
                    with col2:
                        st.plotly_chart(fig_var_2)

            st.markdown("---")





            # Y_2 = np.linspace(0,10,100)
            # X_2 = [f_2(y) for y in Y_2]
            # X_1 = np.linspace(0,arg_max,100)

            

            # Créez un objet trace pour le premier diagramme araignée avec remplissage
            trace_X_1 = go.Scatterpolar(
                r=trans,
                theta=categories,
                fill='toself',
                fillcolor='rgba(0, 0, 128, 0.7)',  # Couleur pour remplir la zone incluse (vert ici)
                name='X'
            )

            # Créez un objet trace pour le deuxième diagramme araignée avec remplissage
            trace_deltaX_1 = go.Scatterpolar(
                r=values_1,
                theta=categories,
                fill='tonext',
                fillcolor='rgba(0, 128, 0, 0.7)',  # Couleur pour remplir la zone incluse (rouge ici)
                name='delta_X'
            )

            # Créez une liste de traces pour les deux diagrammes
            traces_1 = [
            trace_X_1, 
            trace_deltaX_1
            ]

            # Créez la mise en page du diagramme araignée
            layout = go.Layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ), title = 'Modèle 1', width = 650,
                showlegend=True
            )

            # Créez la figure
            fig_X_1 = go.Figure(data=traces_1, layout=layout)


            # Créez un objet trace pour le premier diagramme araignée avec remplissage
            trace_X_2 = go.Scatterpolar(
                r=trans,
                theta=categories,
                fill='toself',
                fillcolor='rgba(0, 0, 128, 0.7)',  # Couleur pour remplir la zone incluse (vert ici)
                name='X'
            )

            # Créez un objet trace pour le deuxième diagramme araignée avec remplissage
            trace_deltaX_2 = go.Scatterpolar(
                r=values_2,
                theta=categories,
                fill='tonext',
                fillcolor='rgba(0, 128, 0, 0.7)',  # Couleur pour remplir la zone incluse (rouge ici)
                name='delta_X'
            )

            # Créez une liste de traces pour les deux diagrammes
            traces_2 = [
            trace_X_2, 
            trace_deltaX_2
            ]

            # Créez la mise en page du diagramme araignée
            layout_2 = go.Layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ), title = 'Modèle 2', width = 650,
                showlegend=True
            )

            # Créez la figure
            fig_X_2 = go.Figure(data=traces_2, layout=layout_2)

            col1, col2 =st.columns(2)
            with col1:
                st.plotly_chart(fig_X_1)
            with col2:
                st.plotly_chart(fig_X_2)

        


#-------------------------------------------------------------------------------------------------------------------------------------
# Bouton saisie
#-------------------------------------------------------------------------------------------------------------------------------------

    
    #st.write(f"Impact: {A_input}")
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col2:
        # Bouton pour valider les saisies
        valider_saisies = st.button("Valider mes saisies")

#-------------------------------------------------------------------------------------------------------------------------------------
# Résultats
#-------------------------------------------------------------------------------------------------------------------------------------


    if valider_saisies:
        st.header(str(compteur_titres) +'- Résultats')
        imp_net = format_number(impact_value-money-envt)
        st.markdown(
        f"<div style='text-align: center; font-size: 24px; font-weight: bold; border: 2px solid black; padding: 10px;'> Impact contrefactuel net = {imp_net}</div>",
        unsafe_allow_html=True)

        if EI:
            imp_EI_net = format_number((impact_EI-money_EI-envt_EI)-(impact_value - money - envt))
            st.markdown(
            f"<div style='text-align: center; font-size: 24px; font-weight: bold; border: 2px solid black; padding: 10px;'> Impact contrefactuel net de EI = {imp_EI_net}</div>",
            unsafe_allow_html=True)



        



        n = np.arange(1,n_tot)
        t = np.linspace(0,horizon -1,horizon, dtype=int)

        imp_gp_vect = np.vectorize(impact_class.fonction_impact_gp, otypes=[np.float32])
        I_n = imp_gp_vect(n)
        

        imp_temps_vect = np.vectorize(impact_class.fonction_impact_temps, otypes=[np.float32])
        I_t = imp_temps_vect(t)

        

        L_temps_gp = []
        for k in range(nb_gp):
            imp_tps_gp = np.vectorize(impact_class.fonction_impact_temps_gp, otypes=[np.float32])
            L_temps_gp.append(imp_tps_gp(k+1, t))


        if EI:
            imp_gp_EI_vect = np.vectorize(impact_class_EI.fonction_impact_gp, otypes=[np.float32])
            I_n_EI = imp_gp_EI_vect(n)
            imp_temps_EI_vect = np.vectorize(impact_class_EI.fonction_impact_temps, otypes=[np.float32])
            I_t_EI = imp_temps_EI_vect(t)
            L_temps_EI_gp = []
            for k in range(nb_gp):
                imp_tps_EI_gp = np.vectorize(impact_class_EI.fonction_impact_temps_gp, otypes=[np.float32])
                L_temps_EI_gp.append(imp_tps_EI_gp(k+1, t))

        # L_temps_gp = []
        # for k in range(nb_gp):
        #     fonction = impact_class.fonction_impact_temps_gp
        #     imp_tps_gp = np.array([fonction(k+1,t_k,Solutions) for t_k in t])
        #     L_temps_gp.append(imp_tps_gp)
        # #st.write(str(L_temps_gp))

        f_WB_1 = impact_class.f_WB
        # def f_WB_real(n,t):
        #     return f_WB_1(n,t, Solutions)


        N, T = np.meshgrid(n, t)
        f_WB_vect = np.vectorize(f_WB_1, otypes=[np.float32])
        I_nt = f_WB_vect(N,T)

        n = pas_indiv*n


        hist = impact_class.fonction_impact_somme_gp()
        if EI:
            hist_EI = impact_class_EI.fonction_impact_somme_gp()

        grid_layout = go.Layout(
            xaxis=dict(gridcolor='gray', gridwidth=1),  # Grille grise plus foncée sur l'axe des x
            yaxis=dict(gridcolor='gray', gridwidth=1)   # Grille grise plus foncée sur l'axe des y
        )

        black_with_opacity = 'rgba(0, 0, 0, 0.4)'

        if EI:
            fig_7 = px.bar(df_imp, x= 'EI', y='Valeur', color='Type Impacts/Coûts', text='Valeur', title='Impacts et coûts',
            color_discrete_sequence = ['green', 'blue', 'red'])
            #fig_7.update_traces(marker=dict(color=df_imp["Type Impacts/Coûts"].map(colors)))
            fig_7.update_coloraxes(showscale=False)
            fig_7.update_traces(texttemplate='%{text:,}')

        else:
            fig_7 = px.bar(df_imp, x= 'Type Impacts/Coûts', y='Valeur', text='Valeur', title='Impacts et coûts')
            fig_7.update_traces(marker=dict(color=df_imp["Type Impacts/Coûts"].map(colors)))
            fig_7.update_coloraxes(showscale=False)
            fig_7.update_traces(texttemplate='%{text:,}')



        # Créer un graphe interactif Plotly
        fig_1 = go.Figure()

        # Ajoutez les données tracées
        
        if EI:
            fig_1.add_trace(go.Scatter(x=n, y=I_n_EI, mode='lines', name='Avec EI', line=dict(color='green', dash='solid')))
        fig_1.add_trace(go.Scatter(x=n, y=I_n, mode='lines', name='Sans EI', line=dict(color='red', dash='solid'), showlegend = EI))
        # Ajoutez les lignes verticales avec légendes
        val_n = 0
        N_pop = impact_class.N_pop()
        palette = px.colors.qualitative.G10
        for n in range(nb_gp):
            color = palette[n % len(palette)]
            fig_1.add_vrect(x0=int(N_pop[n]*pas_indiv), x1= int(N_pop[n+1]*pas_indiv), annotation_text= f"Groupe {n+1}", annotation_position = 'bottom', fillcolor= color, opacity=0.3)
            fig_1.add_vline(x=int(N_pop[n]*pas_indiv), line_dash="dash")


        fig_1.update_xaxes(title_text='Population touchée')
        fig_1.update_yaxes(title_text='Well-being sommé sur le temps')
        fig_1.update_layout(width = 650)

        rge = max(abs(min(hist['Impact'])), abs(max(hist['Impact'])))
        color_range = [-rge, rge]

        fig_6 = px.bar(hist, x='Groupes', y='Impact', title='Distribution des groupes', 
                        color='Impact', color_continuous_scale='RdYlGn', range_color = color_range)
        fig_6.update_coloraxes(showscale=False)
        fig_6.update_layout(width = 650)


        if EI:
            fig_6 = go.Figure()

            fig_6.add_trace(go.Bar(
                x=hist['Groupes'],
                y=hist['Impact'],
                name='Distribution des groupes',
                marker_color='blue'  # Utilisez marker_color pour spécifier la couleur basée sur 'Impact'
            ))

            fig_6.add_trace(go.Bar(
                x=hist_EI['Groupes'],
                y=hist_EI['Impact'],
                name='Distribution des groupes avec EI',
                marker_color='green',  # Utilisez marker_color pour spécifier la couleur basée sur 'Impact'
            ))

            fig_6.update_layout(
                title='Distribution des groupes',
                xaxis=dict(title='Groupes'),
                yaxis=dict(title='Impact'),)
    

#        fig_1.update_layout(
#            xaxis=dict(showgrid=True, gridcolor=black_with_opacity),
#            yaxis=dict(showgrid=True, gridcolor=black_with_opacity)
#        )
        fig_2 = go.Figure()
        palette = px.colors.qualitative.G10
        palette_opp = [f'rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.25)' for color in palette]

        color = palette[0]
        color_opp = palette_opp[0]
        fig_2.add_trace(go.Scatter(x=t, y=I_t, mode='lines', name='WB collectif', line=dict(color=color, dash='dash')))
        if EI:
            fig_2.add_trace(go.Scatter(x=t, y=I_t_EI, mode='lines', name='WB collectif avec EI', line=dict(color=color, dash='solid')))
            fig_2.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]),  # x dans les deux sens pour le remplissage
                            y=np.concatenate([I_t, I_t_EI[::-1]]),  # y1 et y2 dans les deux sens
                            fill='tozerox',  # Remplissage jusqu'à l'axe x
                            fillcolor=color_opp,  # Utilisation de la palette personnalisée
                            line=dict(color=color_opp),
                            showlegend=False)
                            )
        for k in range(1,len(L_temps_gp)+1):
            color = palette[k % len(palette)]
            color_opp = palette_opp[k % len(palette)]
            fig_2.add_trace(go.Scatter(x=t, y=L_temps_gp[k-1], mode='lines', name=f'WB groupe {k+1}', line=dict(color=color, dash='dash')))
            if EI:
                fig_2.add_trace(go.Scatter(x=t, y=L_temps_EI_gp[k-1], mode='lines', name=f'WB groupe {k+1}', line=dict(color=color, dash='solid')))
                fig_2.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]),  # x dans les deux sens pour le remplissage
                            y=np.concatenate([L_temps_gp[k-1], L_temps_EI_gp[k-1][::-1]]),  # y1 et y2 dans les deux sens
                            fill='tozerox',  # Remplissage jusqu'à l'axe x
                            fillcolor=color_opp,  # Utilisation de la palette personnalisée
                            line=dict(color=color_opp),
                            showlegend=False)
                            )
        fig_2.update_xaxes(title_text=f'Temps en {unit_temps}')
        fig_2.update_yaxes(title_text='Well-being collectif')
        fig_2.update_traces(texttemplate='%{text:,}')





        fig_3 = go.Figure(data=[go.Surface(z=I_nt, x=100*N, y=T,  colorscale='RdYlGn')])
        fig_3.update_layout(title='Fonction du WB indiv')
        fig_3.update_scenes(xaxis_title='N', yaxis_title='T', zaxis_title='WB')


        col1, col2, col3 = st.columns([2,6,1])

        with col2:
            st.plotly_chart(fig_7)



        col1, col2 =st.columns(2)

        with col1:
            st.plotly_chart(fig_6)
        with col2:
            st.plotly_chart(fig_1)
  

        col1, col2, col3 = st.columns([2,6,1])


        with col2:

            st.plotly_chart(fig_2)

            st.plotly_chart(fig_3)

        st.markdown("---")






if __name__ == "__main__":
    main()