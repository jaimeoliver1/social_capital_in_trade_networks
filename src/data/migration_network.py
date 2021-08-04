import os
import networkx as nx
import pandas as pd
import numpy as np

class MigrationNetworkCreation:
        

    def __init__(self, year: str, input_filepath: str, output_filepath: str):
        self.year=year
        self.input_filepath=input_filepath
        self.output_filepath=output_filepath

    def oecd_matrix_ingestion(self):
        data_path = os.path.join(self.input_filepath, 'MIG_12082020131505678.csv')
        self.df = pd.read_csv(data_path, low_memory=False, dtype={'Year':str})
        
        self.df = self.df[self.df['Variable'] == 'Inflows of foreign population by nationality']
        
        self.df = self.df[self.df.Year == self.year]

        self.df = self.df[['CO2', 'COU', 'Value']]
        self.df.columns = ['country_from', 'country_to', 'weight']  
        
    def population_etl(self):

        data_path = os.path.join(self.input_filepath, 'API_SL.TLF.TOTL.IN_DS2_en_csv_v2_1929128.csv')
        df_population = pd.read_csv(data_path, skiprows=4)
        df_population.drop(columns=['Country Name','Indicator Name', 'Indicator Code'], inplace=True)

        df_population = df_population.set_index(['Country Code']).stack().reset_index()
        df_population.columns = ['country', 'year','wkn_population']

        self.df_population = df_population[df_population.year == self.year]

        self.df_population = self.df_population[['country','wkn_population']]

    def normalise_by_procedence(self):
        
        self.df =  self.df.merge(self.df_population, how='left', left_on='country_from', right_on='country')
        self.df['weight'] = self.df['weight'].divide(self.df['wkn_population'])
        
        self.df = self.df[['country_from', 'country_to', 'weight']]
        
    def map_row_countries(self):
        
        data_path = os.path.join(self.output_filepath, '2000', 'gdp.parquet')

        df_countries = pd.read_parquet(data_path)

        missing_countries = (set(self.df.country_from) or set(self.df.country_to)) - set(df_countries.index)

        self.df.loc[self.df.country_from.isin(missing_countries) , 'country_from'] = 'ROW'
        self.df.loc[self.df.country_to.isin(missing_countries) , 'country_to'] = 'ROW'

        self.df = self.df.groupby(['country_from', 'country_to']).sum().reset_index()
        
        self.df = self.df.dropna()

    def create_network(self):
    
        self.G = nx.from_pandas_edgelist(self.df, 
                                         source='country_from', 
                                         target='country_to', 
                                         edge_attr='weight',
                                         create_using=nx.DiGraph)

    def run(self):

        self.oecd_matrix_ingestion()
        
        self.population_etl()
        
        self.normalise_by_procedence()

        self.map_row_countries()
        
        self.create_network()

class EstimatedMigrationNetwork:
    
    def __init__(self, B, input_filepath, output_filepath):
        
        self.estimated_M = B.copy()
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        
    def load_emigration_rates(self):

        df_emigration_rate = pd.read_csv(os.path.join(self.input_filepath,'File4_DIOC-E_3_Emigration Rates.csv'), encoding='latin-1')
        columns = ['coub', 'ERT1']
        df_emigration_rate = df_emigration_rate.loc[df_emigration_rate.sex == 'Total', columns]
        df_emigration_rate.columns = ['country', 'emigration_rate']
        df_emigration_rate['emigration_rate'] = df_emigration_rate['emigration_rate']/100
        df_emigration_rate.dropna(inplace=True)
        
        self.emigration_rate = dict(zip(df_emigration_rate.country, df_emigration_rate.emigration_rate))
        
    def estimate_emigration_rate(self):
        
        self.load_emigration_rates()
        
        self.totals = dict(self.estimated_M.out_degree(weight='weight'))
        
        for u,v,d in self.estimated_M.edges(data=True):
            d['weight'] *= self.emigration_rate.get(u, np.nan)/self.totals.get(u, 0)
            if u==v:d['weight'] = 0
                
        return self.estimated_M