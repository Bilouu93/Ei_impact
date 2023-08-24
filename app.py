import streamlit as st
import numpy as np
from modele_mat import *
import random
import plotly.express as px
import plotly.graph_objects as go
from fonction_parametrage import *



L_var = ['Self_control',
    'Evenement_mastery',
    'Time_management_and_anticipation',
    'Goals_in_life',
    'Managing_impact_of_past_experience',
    'Emotional_management',
    'Stress_management',
    'Positive_habits',
    'Efforts_in_life',
    'Selective_secondary_control',
    'No_Hypervigilence',
    'Coping_active',
    'Extraversion',
    'No_Health_locus_of_control',
    'No_Social_potency',
    'Optimism',
    'No_Agreebleness',
    'No_Sympathy',
    'Gratitude',
    'Consciensciousness',
    'Living_the_moment',
    'Self_protection',
    'Physical_attractiveness',
    'Self-efficacy',
    'Openness_to_new_experiences',
    'Compensatory_primary_control',
    'Partenaired',
    'Family_support',
    'No_Famlily_strain',
    'Having_grandparents',
    'Number_of_children',
    'Pregnancy',
    'Positive_relationships',
    'Social_integration',
    'Social_Closeness',
    'No_Tobacco_and_alcohol_of_others',
    'Friends_support',
    'No_Contribution_to_others',
    'Relative_salary',
    'Relative_health',
    'No_Discrimination',
    'Social_coherence',
    'Years_in_state',
    'No_Criminality/Insecurity',
    'Traditionalism',
    'No_Menta_health_professsionnal',
    'Physical_health',
    'No_Sleep_problems',
    'No_Aches',
    'Social_networks',
    'No_Alcohol_problems',
    'No_Drug_consumption',
    'No_Tobacco',
    'Housing_quality',
    'Access_to_local_services',
    'No_No_access_to_water',
    'No_No_access_to_food',
    'No_No_accès_to_health_services',
    'Noise_pollution',
    'Air_and_water_pollution',
    'Close_to_green_spaces',
    'Richessness_fauna,_flaura_and_landscape',
    'Academic_education',
    'Family_education',
    'Physical_activity',
    'Spiritual_and_religious_experiences',
    'Sex_life',
    'Political_participation',
    'Intellectual_activities',
    'Assessment_professionnal_life',
    'Balance_btw_personal_and_professional_life',
    'Assessment_financial_situation',
    'Control_over_professionnal_life']

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

def main():


    st.set_page_config(layout="wide")
    st.image("EI_logo.png", width=200)
    # Utiliser la balise HTML <style> pour centrer le titre
    st.markdown("""
    <style>
        .centered-title {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # Afficher le titre centré
    st.markdown('<h1 class="centered-title">Modèle de prédiction d\'impact</h1>', unsafe_allow_html=True)

    # Définir le style CSS pour les éléments fixes
    style = """ 
        .fixed-author {
            position: fixed;
            bottom: 0;
            left: 0;
            margin: 10px;
            font-style: italic;
        }
    """

    # Appliquer le style
    st.markdown(f'<style>{style}</style>', unsafe_allow_html=True)

    # Afficher l'auteur en bas à gauche (élément fixe)
    st.markdown('<p class="fixed-author">Bilel HATMI</p>', unsafe_allow_html=True)


    #Choix de la dynamique

    st.header("1- Choix de la dynamique")
    options_tau = ['Macro','Micro']
    option_tau = st.radio("Choisir un mode de sélection pour caractériser le régime transitoire:", options_tau)
    bool_indiv = True
    if option_tau == 'Macro':
        tau_global = st.number_input(r"Transitoire  $ \ \tau_{glob}$:", value = 1.0, step=0.5) 
        bool_indiv = False
    horizon = st.number_input(f"Horizon  $ \ T$:", value = 10, step=1)


    #Choix des variables transversales

    st.header("2- Caractérisation des variables transversales")
    Trans = np.zeros(len(L_trans))
    with st.expander("Cliquez ici pour modifier la valeur des variables transversales"):
        col1,col2,col3 = st.columns([5,1,5])
        with col1:
            for n in range(0,int(len(L_trans)/2)):
                t_rd = random.randint(0,100)
                trans_value = st.slider(f"Valeur _{L_trans[n]}_:",value=t_rd, min_value=0, max_value=100, step=1)
                Trans[n] = trans_value -100
        with col3:
            for n in range(int(len(L_trans)/2), len(L_trans)):
                t_rd = random.randint(0,100)
                trans_value = st.slider(f"Valeur _{L_trans[n]}_:",value=t_rd, min_value=0, max_value=100, step=1)
                Trans[n] = trans_value-100
    

    st.header("3- Caractérisation des groupes de population")
    # Demander à l'utilisateur de saisir le nombre de groupes
    nb_gp = st.number_input("Nombre de groupes:", value = 2, min_value=1, step=1)

    A_input = {}
    with st.expander("Cliquez ici pour modifier la valeur des variables prédictives"):

        # Saisie des variables pour chaque groupe
        for group_id in range(1, nb_gp + 1):
            n_rd = random.uniform(10,20)
            A_input[group_id] = [(str(i), 0.0, 0.0) for i in range(len(L_var) + 1)]
            st.write(f"### <b>Groupe {group_id}</b>", unsafe_allow_html=True)
            taille_pop = st.number_input(f"Valeur taille du groupe:", value = int(n_rd), step=10, key=f"group{group_id}_taille" )
            if bool_indiv:
                tau_pop = st.number_input(r"$\tau$ taille du groupe:", value = 0.00, step=0.01, key=f"group{group_id}_pop_tau" )
            else:
                tau_pop = tau_global
            st.markdown("---")
            A_input[group_id][-1] = ('taille', taille_pop, tau_pop)
            col1,col2,col3 = st.columns([5,1,5])
            with col1:
                for n in range(int(len(L_var)/2)+1):
                    if group_id==1:
                        rd_value = random.uniform(-0.2,0.6)
                    elif group_id==2:
                        rd_value = random.uniform(-0.5,0.1)
                    else:
                        rd_value = random.uniform(-0.2,1)
                    variable_value = st.number_input(f"Valeur _{L_var[n]}_:", value = rd_value, step=0.01, key=f"group{group_id}_{L_var[n]}" )
                    #st.markdown(f"Valeur <i>{L_var[n]}</i> ", unsafe_allow_html=True)
                    #variable_value = st.number_input(value = 0.00, step=0.01)
                    if bool_indiv:
                        tau_value = st.number_input(f"$\\tau$ _{L_var[n]}_:", value = 1.0, step=0.5, key=f"group{group_id}_{L_var[n]}_tau" )
                    else:
                        tau_value = tau_global
                    A_input[group_id][n] = (L_var[n], variable_value, tau_value)
                    st.markdown("---")
            with col3:
                for n in range(int(len(L_var)/2) + 1, len(L_var)):
                    if group_id==1:
                        rd_value = random.uniform(-0.2,0.6)
                    elif group_id==2:
                        rd_value = random.uniform(-0.5,0.1)
                    else:
                        rd_value = random.uniform(-0.2,1)
                    variable_value = st.number_input(f"Valeur _{L_var[n]}_:", value = rd_value, step=0.01, key=f"group{group_id}_{L_var[n]}" )
                    #st.markdown(f"Valeur <i>{L_var[n]}</i> ", unsafe_allow_html=True)
                    #variable_value = st.number_input(value = 0.00, step=0.01)
                    if bool_indiv:
                        tau_value = st.number_input(f"Tau _{L_var[n]}_:", value = 1.0, step=0.5, key=f"group{group_id}_{L_var[n]}_tau" )
                    else:
                        tau_value = tau_global
                    A_input[group_id][n] = (L_var[n], variable_value, tau_value)
                    st.markdown("---")
                
    
    impact_class = Impact(A_input,Trans,horizon)
    impact_value = round(impact_class.impact_tot())
    n_tot = impact_class.n_glob()


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
            fig_4.update_xaxes(title_text='K')
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
            fig_5.update_xaxes(title_text='K')
            fig_5.update_yaxes(title_text='p_s')
            fig_5.update_layout(
                xaxis=dict(showgrid=True, gridcolor=black_with_opacity),
                yaxis=dict(showgrid=True, gridcolor=black_with_opacity), width=650
            )
            highlight_trace_5 = px.scatter(x=[Ks_true], y=[ps_true] )
            highlight_trace_5 = go.Scatter(x=[Ks_true], y=[ps_true], mode='markers', marker=dict(color='black', size=10), name='Point actuel')
            fig_5.add_trace(highlight_trace_5)
            st.plotly_chart(fig_5)



    

    
    #st.write(f"Impact: {A_input}")
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col2:
        # Bouton pour valider les saisies
        valider_saisies = st.button("Valider mes saisies")

    if valider_saisies:
        st.header('5) Résultats')
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

        black_with_opacity = 'rgba(0, 0, 0, 0.4)'
        # Créer un graphe interactif Plotly
        fig_1 = px.line(x=n, y=I_n, title='Fonction de WB sommé sur le temps')
        fig_1.update_traces(line=dict(color='red', dash='solid'))
        fig_1.update_xaxes(title_text='Population touchée')
        fig_1.update_yaxes(title_text='Well-being sommé sur le temps')

        fig_1.update_layout(
            xaxis=dict(showgrid=True, gridcolor=black_with_opacity),
            yaxis=dict(showgrid=True, gridcolor=black_with_opacity)
        )

        fig_2 = px.line(x=t, y=I_t, title='Fonction du WB collectif')
        fig_2.update_traces(line=dict(color='blue', dash='solid'))
        fig_2.update_xaxes(title_text='Temps')
        fig_2.update_yaxes(title_text='Well-being collectif')

        fig_2.update_layout(
            xaxis=dict(showgrid=True, gridcolor=black_with_opacity),
            yaxis=dict(showgrid=True, gridcolor=black_with_opacity)
        )

        fig_3 = go.Figure(data=[go.Surface(z=I_nt, x=N, y=T)])
        fig_3.update_layout(title='Fonction du WB indiv')
        fig_3.update_scenes(xaxis_title='N', yaxis_title='T', zaxis_title='WB')


        
        st.plotly_chart(fig_1)

        st.plotly_chart(fig_2)

        st.plotly_chart(fig_3)


if __name__ == "__main__":
    main()