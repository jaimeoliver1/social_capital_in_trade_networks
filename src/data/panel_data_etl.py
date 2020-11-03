import os
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import PowerTransformer

class PanelDataETL:
    
    def __init__(self,input_filepath, output_filepath):
        
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

        self.centralities = ['authorities', 'hubs', 'pagerank', 'betweenness_centrality', 'gfi', 'bridging', 'favor']
        #self.centralities = ['pagerank', 'gfi', 'bridging', 'favor']

    def networks_etl(self):

        all_years = []

        for year in range(2005, 2016):
            
            year = str(year)
            # Capital network --------------------------------------------
            network_path = os.path.join(self.output_filepath, year, 'A_country.graphml')
            G = nx.readwrite.graphml.read_graphml(network_path)

            df = pd.DataFrame(index=G.nodes)

            for c in self.centralities:
                df['financial_'+c] = df.index.map(nx.get_node_attributes(G,c))
                
            df['financial_hhi'] = df.index.map(nx.get_node_attributes(G,'hhi_index'))


            # Migration network ---------------------------------------------
            network_path = os.path.join(self.output_filepath, year, 'migration_network.graphml')
            G = nx.readwrite.graphml.read_graphml(network_path)

            for c in self.centralities:
                df['human_'+c] = df.index.map(nx.get_node_attributes(G,c))

            df['human_hhi'] = df.index.map(nx.get_node_attributes(G,'hhi_index'))
            
            # Compile ---------------------------
            out_path = os.path.join(self.output_filepath, year, 'industry_output.parquet')
            df_out=pd.read_parquet(out_path)
            df_year = df.merge(df_out, left_index=True, right_index=True)

            gdp_path = os.path.join(self.output_filepath, year, 'gdp.parquet')
            df_gdp=pd.read_parquet(gdp_path)
            df_year = df_year.merge(df_gdp, left_index=True, right_index=True)
            
            df_year['year'] = year
            
            all_years.append(df_year)

        df = pd.concat(all_years)

        self.df = df.reset_index().rename(columns={'index':'country', 'OUTPUT':'output'})


    @staticmethod
    def power_tansformation(df, columns):

        pt = PowerTransformer()

        df[columns] = pt.fit_transform(df[columns])
        
        return df

    def feature_computation(self):

        self.df['log_output'] = np.log(self.df['output'] + 1)
        self.df['log_gdp'] = np.log(self.df['gdp'] + 1)
        
        self.df = self.df.sort_values(by=['country', 'year'])


        all_centrality_cols = ['financial_'+c for c in self.centralities]
        all_centrality_cols += ['human_'+c for c in self.centralities]
        
        for c in all_centrality_cols + ['log_output', 'log_gdp']:
            self.df['lag_' + c] = self.df.groupby('country')[c].shift(1)
            self.df['delta_' + c] = self.df[c] - self.df['lag_' + c]
            self.df['per_change_' + c] = self.df['delta_' + c]/self.df['lag_' + c]

        self.df['lag_log2_output'] = self.df.groupby('country').log_output.shift(2)
        self.df['lag_log2_gdp'] = self.df.groupby('country').log_gdp.shift(2)
        
        
        self.df = self.power_tansformation(df = self.df, columns = all_centrality_cols)
        
        return self.df

    def run_one_year_gross_capital_formation(self, year):
    
        data_path = os.path.join(self.input_filepath, f'ICIO2018_{year}.zip')
        df = pd.read_csv(data_path, compression='zip'
                    ).set_index("Unnamed: 0")
        df = df[[c for c in df.columns if 'GFCF' in c]]

        df_totals = df.sum().T
        df_totals = pd.DataFrame(df_totals).reset_index()
        df_totals.columns = ['country', 'GFCF']
        
        df_totals['country'] = df_totals['country'].map(lambda x: x[:3])
        df_totals['log_GFCF'] = df_totals['GFCF'].map(lambda x: np.log(x + 1))
            
        df_totals['year'] = str(year)
        
        return df_totals
        
    def get_all_years_gross_capital_formation(self):
    
        all_years = []
        for y in range(2005, 2016):
            df_year = self.run_one_year_gross_capital_formation(year=y)
            
            all_years.append(df_year)
            
        
        df = pd.concat(all_years, axis=0)
        
        
        # Compute lags and deltas
        df = df.sort_values(by=['country', 'year'])
        df['lag_log_GFCF'] = df.groupby('country').log_GFCF.shift(1)
        df['delta_log_GFCF'] = df['log_GFCF'] - df['lag_log_GFCF']

        return df
        
    def population_etl(self):

        data_path = os.path.join(self.input_filepath, 'DP_LIVE_06072020184943320.csv')
        df_working = pd.read_csv(data_path, dtype={'TIME':str})
        df_working.rename(columns = {'LOCATION':'country','TIME':'year', 'Value':'pctg'}, inplace=True)
        df_working = df_working[['country', 'year', 'pctg']]

        data_path = os.path.join(self.input_filepath, 'DP_LIVE_06072020200357239.csv')
        df_population = pd.read_csv(data_path, dtype={'TIME':str})

        df_population = df_population[df_population.MEASURE == 'MLN_PER']
        df_population = df_population[df_population.SUBJECT == 'TOT']

        df_population = df_population[['LOCATION', 'TIME', 'Value']]

        df_population.columns = ['country', 'year', 'population']


        df_population = df_population.merge(df_working, how='left', on = ['country', 'year'])

        df_population['wkn_population'] = df_population['population']*df_population['pctg']

        df_population['log_population'] = df_population['population'].map(lambda x: np.log(x + 1))
        df_population['log_wkn_population'] = df_population['wkn_population'].map(lambda x: np.log(x + 1))

        # Compute lags and deltas
        df_population = df_population.sort_values(by=['country', 'year'])
        df_population['lag_log_population'] = df_population.groupby('country').log_population.shift(1)
        df_population['delta_log_population'] = df_population['log_population'] - df_population['lag_log_population']
        df_population['lag_log_wkn_population'] = df_population.groupby('country').log_wkn_population.shift(1)
        df_population['delta_log_wkn_population'] = df_population['log_wkn_population'] - df_population['lag_log_wkn_population']

        return df_population

    def gini_etl(self):

        data_path = os.path.join(self.input_filepath, 'DP_LIVE_13102020161705689.csv')
        df_gini = pd.read_csv(data_path, dtype={'TIME':str})
        
        df_gini = df_gini[df_gini.SUBJECT == 'GINI']

        df_gini = df_gini[['LOCATION', 'TIME', 'Value']]
        df_gini.columns = ['country', 'year', 'gini']

        return df_gini

    def run(self):

        self.networks_etl()
        df_net = self.feature_computation()
        df_gfcf =  self.get_all_years_gross_capital_formation()
        df_model = df_net.merge(df_gfcf, how='left')
    
        df_population = self.population_etl()
        df_model = df_population.merge(df_model, how='right', left_on=['country', 'year'], right_on = ['country', 'year'])
        
        df_gini = self.gini_etl()
        df_model = df_gini.merge(df_model, how='right', left_on=['country', 'year'], right_on = ['country', 'year'])

        df_model.dropna(subset = ['population'], inplace=True)
        df_model['constant'] = 1

        df_model.year = df_model.year.astype(int)
        df_model.query('year <= 2015 & year >=2005', inplace=True)

        print('countries lost: ', set(df_net.country) - set(df_population.country))
        return df_model
