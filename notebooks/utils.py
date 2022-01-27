import pandas as pd
import os
import numpy as np

def data_loader(output_filepath):
    
    data_path = os.path.join(output_filepath, 'panel_data.parquet')
    df_model = pd.read_parquet(data_path)

    df_model = df_model[df_model.year.between(1990, 2016)]

    centralities = ['hubs', 'authorities','pagerank', 'gfi', 'bridging', 'favor']
    centralities = ['hubs', 'authorities', 'bridging', 'favor']
    centralities = ['hubs', 'authorities', 'favor']
    networks = ['financial', 'goods', 'human']

    #df_model = df_model[~df_model.country.isin(['ETH', 'BLR', 'ZWE', 'MDA', 'GUY', 'VNM', 'MAC', 'PSE', 'AGO', 'COD', 'TZA'])]

    df_model.dropna(subset=['log_GFCF'], inplace=True)
    df_model = df_model[(df_model.log_GFCF>0)&(df_model.log_gdp>0)&(~df_model.financial_hubs.isnull())]

    df_model.eval('productivity = gdp*10**6/(wkn_population**0.3*GFCF**0.7)', inplace=True)
    for c in [f'{n}_{c}' for c in centralities for n in networks] + ['productivity']:
        df_model[c] = df_model[c].map(lambda x: np.log1p(x*1.e8))

    #df_model = df_model[df_model.wkn_population>5*1.e6]

    all_terms_list = [f'{n}_{c}' for n in networks for c in centralities]
    reduced_terms_list = all_terms_list.copy()
    reduced_terms_list.remove('goods_favor')
    reduced_terms_list.remove('financial_favor')
    reduced_terms_list.remove('human_authorities')

    df_model.sort_values(by = ['country', 'year'], inplace=True)    

    return reduced_terms_list, df_model