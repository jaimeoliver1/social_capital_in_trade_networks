import networkx as nx
import pandas as pd

class MigrationNetworkCreation:
        

    def __init__(self, year=2015):
        self.year=year
        
    def oecd_matrix_ingestion(self):
        
        self.df = pd.read_csv('s3://workspaces-clarity-mgmt-pro/jaime.oliver/jobs/value_chain/oecd/migration/MIG_12082020131505678.csv', low_memory=False)
        
        self.df = self.df[self.df['Variable'] == 'Inflows of foreign population by nationality']
        
        self.df = self.df[self.df.Year == self.year]

        self.df = self.df[['CO2', 'COU', 'Value']]
        self.df.columns = ['country_from', 'country_to', 'weight']  
        
    def population_etl(self):
        
        df_working = pd.read_csv('s3://workspaces-clarity-mgmt-pro/jaime.oliver/jobs/value_chain/oecd/DP_LIVE_06072020184943320.csv')
        df_working.rename(columns = {'LOCATION':'country','TIME':'year', 'Value':'pctg'}, inplace=True)
        df_working = df_working[['country', 'year', 'pctg']]
        df_working = df_working[df_working.year == self.year]

        self.df_population = pd.read_csv('s3://workspaces-clarity-mgmt-pro/jaime.oliver/jobs/value_chain/oecd/DP_LIVE_06072020200357239.csv')

        self.df_population = self.df_population[self.df_population.MEASURE == 'MLN_PER']
        self.df_population['Value'] = self.df_population['Value']*1e+6
        
        self.df_population = self.df_population[self.df_population.SUBJECT == 'TOT']

        self.df_population = self.df_population[self.df_population.TIME == self.year]
        self.df_population = self.df_population[['LOCATION', 'Value']]

        self.df_population.columns = ['country', 'population']

        self.df_population = self.df_population.merge(df_working, how='left', on = ['country'])

        self.df_population['wkn_population'] = self.df_population['population']*self.df_population['pctg']
        self.df_population = self.df_population[['country', 'wkn_population']]


    def normalise_by_procedence(self):
        
        self.df =  self.df.merge(self.df_population, how='left', left_on='country_from', right_on='country')
        self.df['weight'] = self.df['weight'].divide(self.df['wkn_population'])
        
        self.df = self.df[['country_from', 'country_to', 'weight']]
        
    def map_row_countries(self):
        
        df_countries = pd.read_parquet('///domino/datasets/jaime_oliver/industry_network/scratch/2015/gdp.parquet')

        missing_countries = (set(self.df.country_from) or set(self.df.country_to)) - set(df_countries.index)

        self.df.loc[self.df.country_from.isin(missing_countries) , 'country_from'] = 'ROW'
        self.df.loc[self.df.country_to.isin(missing_countries) , 'country_to'] = 'ROW'

        self.df = self.df.groupby(['country_from', 'country_to']).sum().reset_index()
        
        #self.df['weight'] = self.df['weight'].astype(int)
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