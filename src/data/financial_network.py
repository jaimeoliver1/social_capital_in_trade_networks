from dask.distributed import Client
import dask.array as da

import pandas as pd
import numpy as np

import datetime


class IndustryNetworkCreation:

    def __init__(self, year=2015):
        self.year=year

    def oecd_matrix_ingestion(self, by_country=False):
        # Read data
        df = pd.read_csv(f"s3://workspaces-clarity-mgmt-pro/jaime.oliver/jobs/value_chain/oecd/input_output/ICIO2018_{self.year}.zip", compression='zip'
        ).set_index("Unnamed: 0")
        demand_vars = ["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "P33"]
        supply_vars = ["TAXSUB", "VALU", "OUTPUT","TOTAL"]
        
        # Aggregate Mexico and China
        agg_dict = {"MX1": "MEX", "MX2": "MEX", "CN1": "CHN", "CN2": "CHN"}

        new_columns = df.columns
        new_index = df.index
        for k, v in agg_dict.items():
            new_columns = [c.replace(k, v) for c in new_columns]
            new_index = [c.replace(k, v) for c in new_index]

        if by_country:
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
        
        # Leontief inverse matrix
        self.L = np.linalg.inv(np.eye(len(self.A)) - self.A) 

        # value added per unit output -- 0 value added if no value added is computed
        self.value_added_per_output_unit =  self.w / self.x

        # Final industry contribution to demand
        self.value_adjusted_L = np.dot(np.diag(self.value_added_per_output_unit), self.L)

    def downstream_chain(self):
        
        # Normalisation by outputs (rows) + consumption
        self.B = self.Z/self.x[:,None]
        
        # Ghosh inverse matrix
        self.G_T = np.linalg.inv(np.eye(len(self.B)) - self.B.T) 

        # Consumption per unit output
        self.consumption_per_output_unit =  self.f / self.x

    def save_to_s3(self, execution_date, save_path = 's3://workspaces-clarity-mgmt-pro/jaime.oliver/jobs/value_chain/'):
        ############################################################################################
        # Save absorbing markov chain in format: absorbtion probabilities, transition matrix
        ############################################################################################
        # Outputs -------------------------------
        df_output = pd.DataFrame({'output':self.x}, index=self.node_index)
        df_output.to_parquet(f"{save_path}{execution_date}/industry_output.parquet")

        # GDP
        df_gdp = pd.DataFrame({'gdp':self.w}, index=self.node_index)
        df_gdp.to_parquet(f"{save_path}{execution_date}/gdp.parquet")
        
        # Upstream chain -------------------------------------------------
        # Absobtion probabilities
        df_value_added = pd.DataFrame(
            {"value_added": self.value_added_per_output_unit}, index=self.node_index
        )
        df_value_added.to_parquet(f"{save_path}{execution_date}/industry_value_added.parquet")
        
        # Transition matrix
        df_A = pd.DataFrame(
            self.A, columns=self.node_index, index=self.node_index
        )
        df_A.to_parquet(f"{save_path}{execution_date}/industry_A.parquet")

        # Final industry contribution to demand
        df_L = pd.DataFrame(self.value_adjusted_L, columns=self.node_index, index=self.node_index)
        df_L.to_parquet(f"{save_path}{execution_date}/value_adjusted_L.parquet")

        # Downstream chain -------------------------------------------------
        # Absobtion probabilities
        df_consumption = pd.DataFrame(
            {"value_added": self.consumption_per_output_unit}, index=self.node_index
        )
        df_consumption.to_parquet(f"{save_path}{execution_date}/industry_consumption.parquet")
        
        # Transition matrix
        df_B = pd.DataFrame(
            self.B.T, columns=self.node_index, index=self.node_index
        )
        df_B.to_parquet(f"{save_path}{execution_date}/industry_B.parquet")
        
    def run(self):

        self.oecd_matrix_ingestion()

        self.upstream_chain()
        
        self.downstream_chain()

        self.save_to_s3(execution_date=datetime.date.today())
