import os

from pathlib import Path

import pandas as pd
import numpy as np

import datetime


class IndustryNetworkCreationEORA:

    def __init__(self, year: str, input_filepath: str, output_filepath: str):

        if year == '2016':
            self.year='2015'
        else:
            self.year=year
            
        self.input_filepath='s3://workspaces-clarity-mgmt-pro/jaime.oliver/misc/EORA/'
        self.output_filepath=output_filepath

    def eora_matrix_ingestion(self):

        self.data_path = os.path.join(self.input_filepath, f'Eora26_{self.year}_bp/')

        self.df_labels = pd.read_table(os.path.join(self.data_path, 'labels_T.txt') , header=None)
        self.df_labels.columns = ['country_name', 'country', 'type', 'industry', 'drop']

        self.df_T = pd.read_table(os.path.join(self.data_path, f'Eora26_{self.year}_bp_T.txt'), header=None)
        self.df_T.index = self.df_labels.country
        self.df_T.columns = self.df_labels.country

        self.w = pd.read_table(os.path.join(self.data_path, f'Eora26_{self.year}_bp_VA.txt'), header=None)
        self.w = self.w.sum(axis=0)
        self.w = self.w.to_frame()
        self.w.index = self.df_labels.country
        self.w.columns = ['value_added']

    def aggregate_by_country(self):

        self.df_T = self.df_T.groupby(axis=1, level=0).sum()
        self.df_T = self.df_T.groupby(level=0).sum()

        self.w = self.w.groupby(by='country').sum()

        self.node_index = self.df_T.index

    def upstream_chain(self):

        df_A = pd.concat([self.df_T, self.w.T]).apply(lambda x: x/x.sum())
        self.A = df_A[:-1].values

    def get_output(self):

        x = pd.concat([self.df_T, self.w.T]).sum(axis=0)
        self.df_output = pd.DataFrame({'OUTPUT':x}, index = self.node_index)
        
    def get_gdp(self):

        self.df_gdp = self.w.copy()
        self.df_gdp.columns = ['gdp']

    def run(self):

        self.eora_matrix_ingestion()

        self.aggregate_by_country()

        self.upstream_chain()

        self.get_output()
        
        self.get_gdp()

class IndustryNetworkCreation:

    def __init__(self, year: str, input_filepath: str, output_filepath: str):
        self.year=year
        self.input_filepath=input_filepath
        self.output_filepath=output_filepath

    def oecd_matrix_ingestion(self):
        # Read data
        data_path = os.path.join(self.input_filepath,f'ICIO2018_{self.year}.zip')
        df = pd.read_csv(data_path, compression='zip').set_index("Unnamed: 0")
        demand_vars = ["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "P33"]
        supply_vars = ["TAXSUB", "VALU", "OUTPUT","TOTAL"]
        
        # Aggregate Mexico and China
        agg_dict = {"MX1": "MEX", "MX2": "MEX", "CN1": "CHN", "CN2": "CHN"}

        new_columns = df.columns
        new_index = df.index
        for k, v in agg_dict.items():
            new_columns = [c.replace(k, v) for c in new_columns]
            new_index = [c.replace(k, v) for c in new_index]

        new_columns = [c[:3] if c.split('_')[-1] not in demand_vars + supply_vars else c for c in new_columns]
        new_index = [c[:3] if c.split('_')[-1] not in demand_vars + supply_vars else c for c in new_index]

        df.columns = new_columns
        df.index = new_index

        df = df.groupby(axis=1, level=0).sum()
        df = df.groupby(level=0).sum()

        # Keep final demand appart
        final_demand = [c for c in df.columns if c[4:] in demand_vars]
        df_final_demand = df[final_demand]
        df_final_demand = df_final_demand[
            ~df.index.str.contains("|".join(supply_vars))
        ]
        df_final_demand = df_final_demand.sum(axis=1)
        
        df.drop(columns=final_demand, inplace=True)

        # Keep taxes and value added appart
        df_tax = df[df.index.str.contains("TAXSUB")].drop(columns="TOTAL")
        df_value_added = df[df.index == "VALU"].drop(columns="TOTAL")
        df = df[~df.index.str.contains("|".join(supply_vars))]

        # Keep totals appart
        outputs = df["TOTAL"]
        df_out = df["TOTAL"].reset_index()
        df_out.columns = ["country_industry", "OUTPUT"]
        df_out["OUTPUT"] = df_out["OUTPUT"] * 1e6
        
        ###############################################
        # Ad-hoc remove industries with no ouput
        ###############################################
        zero_output_industries = ['AUS_97T98', 'CHL_97T98', 'CHN_97T98', 'EST_97T98', 'JPN_97T98','KHM_97T98', 'NZL_97T98', 'PHL_97T98', 'SGP_05T06', 'SGP_07T08','SGP_09']
        zero_output_industries = df_out[df_out.OUTPUT == 0].country_industry
        
        # Ouput
        self.x = outputs[~outputs.index.isin(zero_output_industries)].values

        # Value added
        df_value_added = df_value_added.T[~df_value_added.columns.isin(zero_output_industries)].T
        df_tax = df_tax.T[~df_tax.columns.isin(zero_output_industries)].T
        self.w = df_value_added.values[0] + df_tax.sum(axis=0).values

        # Final consumption
        self.f = df_final_demand[~df_final_demand.index.isin(zero_output_industries)].values
        
        # Input-output matrix
        df.drop(columns = zero_output_industries + ["TOTAL"], inplace=True, errors='ignore')
        df = df[~df.index.isin(zero_output_industries)]
        self.Z = df[df.index].values

        self.node_index = df.index

    def upstream_chain(self):
        # Normalisation by inputs (columns) + value added
        self.A = self.Z/self.x
        
        # value added per unit output -- 0 value added if no value added is computed
        self.value_added_per_output_unit =  self.w / self.x

    def save(self):
        ############################################################################################
        # Save absorbing markov chain in format: absorbtion probabilities, transition matrix
        ############################################################################################
        # Outputs -------------------------------
        data_path = os.path.join(self.output_filepath, self.year,'industry_output.parquet')
        df_output = pd.DataFrame({'output':self.x}, index=self.node_index)
        df_output.to_parquet(data_path)

        # GDP
        data_path = os.path.join(self.output_filepath, self.year,'gdp.parquet')
        df_gdp = pd.DataFrame({'gdp':self.w}, index=self.node_index)
        df_gdp.to_parquet(data_path)
        
        # Upstream chain -------------------------------------------------
        # Absobtion probabilities
        data_path = os.path.join(self.output_filepath, self.year,'industry_value_added.parquet')
        df_value_added = pd.DataFrame(
            {"value_added": self.value_added_per_output_unit}, index=self.node_index
        )
        df_value_added.to_parquet(data_path)
    
    def get_output(self):

        self.df_output = pd.DataFrame({'OUTPUT':self.x}, index=self.node_index)

    def get_gdp(self):

        self.df_gdp = pd.DataFrame({'gdp':self.w}, index=self.node_index)

    def run(self):

        self.oecd_matrix_ingestion()

        self.upstream_chain()

        self.get_output()

        self.get_gdp()