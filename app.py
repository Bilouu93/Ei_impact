import streamlit as st
import numpy as np
from modele_mat import *
import random
import plotly.express as px
import plotly.graph_objects as go
from fonction_parametrage import *
import pickle
from Importation import B_input, Trans_AB, C_grp_input



L_var = list(dic_wb.keys())

L_trans = ['Non Menace nouveaux entrants',
    'Non Produit de substitution',
    'Non Intensité concurrentielle',
    'Non Pouvoir des clients',
    'Non Pouvoir des fournisseurs',
    'Expertise domaine',
    'Réactivité',
    'Gestion temps/risque',
    "Contrôle chaine d'approvisionnement",
    'Ressources financières',
    'Réseaux/partenaires',
    'Caractère innovant',
    'SAV/communication clients',
    'Cohérence du postionnement']

dic_pas = {'Années':1, 'Semestre':0.5, 'Trimestre':1/3, 'Mois':1/12, 'Jour':1/365}


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

    st.header("1- Fenêtre de temps")

    unit_temps = ['Années','Semestre', 'Trimestre', 'Mois', 'Jour']
    unit_temps = st.radio("Choisir un mode de discrétisation:", unit_temps)
    pas = dic_pas[unit_temps]






    # bool_indiv = True
    # if option_tau == 'Macro':
    #     tau_global = st.number_input(r"Transitoire  $ \ \tau_{glob}$:", value = 1.0, step=0.1) 
    #     bool_indiv = False
    horizon = st.number_input(f"Taille de la fenêtre  $ \ T$:", value = 10, step=1)

    Temps_calcul = st.number_input(r"Temps estimation projet  $ \ T_{projet}$:", value = 10, step=1)

#-------------------------------------------------------------------------------------------------------------------------------------
#Choix des variables transversales
#-------------------------------------------------------------------------------------------------------------------------------------


    st.header("2- Caractérisation des variables transversales")
    Trans = np.zeros(len(L_trans))
    with st.expander("Cliquez ici pour modifier la valeur des variables transversales"):
        col1,col2,col3 = st.columns([5,1,5])
        with col1:
            for n in range(0,int(len(L_trans)/2)):
                t_rd = random.randint(0,100)
                trans_value = st.slider(f"Valeur _{L_trans[n]}_:",
                                        #value=t_rd, 
                                        min_value=0, max_value=100, step=1)
                Trans[n] = trans_value -100
        with col3:
            for n in range(int(len(L_trans)/2), len(L_trans)):
                t_rd = random.randint(0,100)
                trans_value = st.slider(f"Valeur _{L_trans[n]}_:",
                                        #value=t_rd, 
                                        min_value=0, max_value=100, step=1)
                Trans[n] = trans_value-100

#-------------------------------------------------------------------------------------------------------------------------------------
#Gropues de population
#-------------------------------------------------------------------------------------------------------------------------------------

    st.header("3- Caractérisation des groupes de population")
    # Demander à l'utilisateur de saisir le nombre de groupes
    nb_gp = st.number_input("Nombre de groupes:", value = 2, min_value=1, step=1)

    A_input = {}
    N_pop = []

    options_tau = ['Macro','Micro']
    option_tau = st.radio("Caractérisez le régime transitoire des variables spécifiques:", options_tau)
    # Saisie des variables pour chaque groupe
    for group_id in range(1, nb_gp + 1):
        n_rd = random.uniform(10,20)
        A_input[group_id] = [(str(i), 0.0, 0.0) for i in range(len(L_var) + 1)]


        st.write(f"## <b>Groupe {group_id}</b>", unsafe_allow_html=True)
        with st.expander("Cliquez ici pour modifier la valeur des variables prédictives"):
            st.write(f"### <b>Tailles de population</b>", unsafe_allow_html=True)
            taille_pop = st.number_input(f"$N_{group_id}$ Taille du groupe:", 
                                            value = int(n_rd), 
                                            step=10, key=f"group{group_id}_taille" )

            tau_pop_glob = st.number_input(r"$\tau_{n_{glob}}$ Croissance:", 
                                        value = 1.00, 
                                        step=0.01, key=f"group{group_id}_pop_tau_1" )

            a_pop = st.number_input(r"$a_{n_{glob}}$ Croissance:", 
                                        value = 0.1, 
                                        step=0.01, key=f"group{group_id}_a_pop" )

            b_pop = st.number_input(r"$b_{n_{glob}}$ Déclin:", 
                            value = 0.1, 
                            step=0.01, key=f"group{group_id}_b_pop" )


            ordres = ['FIFO','FILO']
            ordre = st.radio("Choisir un mode de sélection pour caractériser le régime transitoire:", ordres, key=f"group{group_id}_ordre")
            if ordre == 'FIFO':
                FIFO = True
            else:
                FIFO = False


            st.markdown("---")
            A_input[group_id][-1] = ('taille', taille_pop, tau_pop_glob, a_pop, b_pop, FIFO)




            if option_tau == 'Macro':
                st.write(f"### <b>Dynamique macro des variables prédictives</b>", unsafe_allow_html=True)
                fraction_K1_glob = st.number_input(f"$K_1$ Phase d'adaptation (fraction RP):", 
                                                value = 0, 
                                                step=10, key=f"group{group_id}_K1" )

                tau_glob = st.number_input(r"$\tau_{glob}$ Croissance:", 
                                            value = 1.00, 
                                            step=0.01, key=f"group{group_id}_tau_glob" )

                a_glob = st.number_input(r"$a_{glob}$ Croissance:", 
                                            value = 0.1, 
                                            step=0.01, key=f"group{group_id}_a_glob" )

                b_glob = st.number_input(r"$b_{glob}$ Déclin:", 
                                value = 0.1, 
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
                    A_input[group_id][n] = (L_var[n], fraction_K1*variable_value, variable_value, a, b)
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
                    A_input[group_id][n] = (L_var[n], fraction_K1*variable_value, variable_value, a, b)
                    st.markdown("---")
                
    
    #st.write(str(A_input))
    #A_input = B_input

    A_input = C_grp_input
    Trans = Trans_AB
    impact_class = Impact(A_input,Trans,horizon, unit_temps)
    impact_value = round(impact_class.impact_tot())
    n_tot = impact_class.n_glob()


#-------------------------------------------------------------------------------------------------------------------------------------
#Vérification paramétrage:
#-------------------------------------------------------------------------------------------------------------------------------------


    st.header('4- Vérification du paramétrage de $k_t$ et $p_s$')
    with st.expander("Cliquez ici pour vérifier le parametrage des fonction $k_t$ et $p_s$"):
        Kt_list = np.linspace(-w_t,0,100)
        Ks_list = np.linspace(-w_s,0,100)
        KT = [kt_2(K) for K in Kt_list]
        PS = [p_s(K) for K in Ks_list]
        kt_true = impact_class.kt_2()
        ps_true = impact_class.ps()
        Kt_true = impact_class.Kt()
        Ks_true = impact_class.Ks()


        col1, col2 = st.columns(2)
        black_with_opacity = 'rgba(0, 0, 0, 0.4)'

        with col1:
            # Créer un graphe interactif Plotly
            fig_4 = px.line(x=Kt_list, y=KT, title='Fonction de pénalisation contrefactuelle')
            fig_4.update_traces(line=dict(color='green', dash='solid'))
            fig_4.update_xaxes(title_text='K_kt')
            fig_4.update_yaxes(title_text='k_t')
            fig_4.update_layout(
                xaxis=dict(showgrid=True, gridcolor=black_with_opacity),
                yaxis=dict(showgrid=True, gridcolor=black_with_opacity), width =650
            )
            highlight_trace_4 = go.Scatter(x=[Kt_true], y=[kt_true], mode='markers', marker=dict(color='black', size=10), name ='Point actuel')
            fig_4.add_trace(highlight_trace_4)
            st.plotly_chart(fig_4)


        with col2:
            # Créer un graphe interactif Plotly
            fig_5 = px.line(x=Ks_list, y=PS, title='Fonction de probabilité de succès')
            fig_5.update_traces(line=dict(color='blue', dash='solid'))
            fig_5.update_xaxes(title_text='K_ps')
            fig_5.update_yaxes(title_text='p_s')
            fig_5.update_layout(
                xaxis=dict(showgrid=True, gridcolor=black_with_opacity),
                yaxis=dict(showgrid=True, gridcolor=black_with_opacity), width=650
            )
            #highlight_trace_5 = px.scatter(x=[Ks_true], y=[ps_true] )
            highlight_trace_5 = go.Scatter(x=[Ks_true], y=[ps_true], mode='markers', marker=dict(color='black', size=10), name='Point actuel')
            fig_5.add_trace(highlight_trace_5)
            st.plotly_chart(fig_5)



    
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
        st.header('5- Résultats')
        #st.write(str(A_input))
        st.markdown(
        f"<div style='text-align: center; font-size: 24px; font-weight: bold; border: 2px solid black; padding: 10px;'> Impact = {impact_value}</div>",
        unsafe_allow_html=True)

        n = np.linspace(0, n_tot, 100)
        t = np.linspace(0,horizon -1,horizon, dtype=int)

        imp_gp_vect = np.vectorize(impact_class.fonction_impact_gp)
        I_n = imp_gp_vect(n)

        imp_temps_vect = np.vectorize(impact_class.fonction_impact_temps)
        I_t = imp_temps_vect(t)

        imp_vectorized = np.vectorize(impact_class.fonction_WB_inf)
        N, T = np.meshgrid(n, t)
        I_nt = imp_vectorized(N,T)

        hist = impact_class.fonction_impact_somme_gp()

        black_with_opacity = 'rgba(0, 0, 0, 0.4)'
        # Créer un graphe interactif Plotly
        fig_1 = go.Figure()

        # Ajoutez les données tracées
        fig_1.add_trace(go.Scatter(x=n, y=I_n, mode='lines', name='Données', line=dict(color='red', dash='solid')))

        # Ajoutez les lignes verticales avec légendes
        val_n = 0
        N_pop = impact_class.N_pop()
        for n in range(nb_gp):
            val_n += N_pop[n]
            Y = [min(I_n), max(I_n)]

            fig_1.add_shape(go.layout.Shape(
                type='line',
                x0=val_n,
                x1=val_n,
                y0=Y[0],
                y1=Y[1],
                line=dict(color='black', width=2, dash='dash'),
                name=f"groupe {n + 1}"
            ))


        fig_1.update_xaxes(title_text='Population touchée')
        fig_1.update_yaxes(title_text='Well-being sommé sur le temps')
        fig_1.update_layout(width = 650)

        rge = max(abs(min(hist['Impact'])), abs(max(hist['Impact'])))
        color_range = [-rge, rge]
        #color_scale = [(1.0, 'red'), (0.0, 'green')]
        fig_6 = px.bar(hist, x='Groupes', y='Impact', title='Distribution des groupes', 
                        color='Impact', color_continuous_scale='RdYlGn', range_color = color_range)
        fig_6.update_coloraxes(showscale=False)


        fig_6.update_layout(width = 650)

#        fig_1.update_layout(
#            xaxis=dict(showgrid=True, gridcolor=black_with_opacity),
#            yaxis=dict(showgrid=True, gridcolor=black_with_opacity)
#        )

        fig_2 = px.line(x=t, y=I_t, title='Fonction du WB collectif')
        fig_2.update_traces(line=dict(color='blue', dash='solid'))
        fig_2.update_xaxes(title_text=f'Temps en {unit_temps}')
        fig_2.update_yaxes(title_text='Well-being collectif')

        fig_2.update_layout(
            xaxis=dict(showgrid=True, gridcolor=black_with_opacity),
            yaxis=dict(showgrid=True, gridcolor=black_with_opacity)
        )

        fig_3 = go.Figure(data=[go.Surface(z=I_nt, x=N, y=T,  colorscale='RdYlGn')])
        fig_3.update_layout(title='Fonction du WB indiv')
        fig_3.update_scenes(xaxis_title='N', yaxis_title='T', zaxis_title='WB')


        col1, col2 =st.columns(2)

        with col1:
            st.plotly_chart(fig_6)
        with col2:
            st.plotly_chart(fig_1)

        col1, col2, col3 = st.columns([2,6,1])

        with col2:
            st.plotly_chart(fig_2)

            st.plotly_chart(fig_3)





if __name__ == "__main__":
    main()