import pandas as pd

file_wb = './values_complet.csv'
df_wb = pd.read_csv(file_wb)
file_ts = './value_transversal.csv'
df_ts = pd.read_csv(file_ts)

#traitement des variables à coef négatifs:
slice = df_wb['Valeurs'] < 0
df_wb.loc[slice, 'Variables'] = 'No_' + df_wb['Variables']
df_wb.loc[slice, 'Valeurs'] = df_wb['Valeurs'].abs()

# Création d'un dictionnaire variable : beta_variable
df_to_dic = df_wb.set_index('Variables')
dic_wb = df_to_dic['Valeurs'].to_dict()