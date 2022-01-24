# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import networkx as nx

from src.utils.utils_features import NetworkFeatureComputation
from src.data.financial_network import (
    IndustryNetworkCreation,
    IndustryNetworkCreationEORA,
)
from src.data.migration_network import MigrationNetworkCreation, EstimatedMigrationNetwork
from src.data.panel_data_etl import PanelDataETL

from src.utils.utils_s3 import read_s3_graphml, write_s3_graphml


def network_from_adjacency(adjacency_matrix, 
                           node_index, 
                           path, 
                           tol_gfi=0.01, 
                           tol_favor=0.0001):
        df_adj = pd.DataFrame(adjacency_matrix, index=node_index, columns=node_index)
        G = nx.convert_matrix.from_pandas_adjacency(df_adj, create_using=nx.DiGraph)
 
        # Compute network features ------------------
        NFC = NetworkFeatureComputation(G)
        NFC.compute_features(tol_gfi=tol_gfi, tol_favor=tol_favor)
        G = NFC.G

        # Save
        write_s3_graphml(G, path)

@click.command()
@click.argument("input_filepath")
@click.argument("output_filepath")
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    
    
    for year in range(2005, 2016):
        
        year = str(year)
        print("Processing year ", year)
        
        # Capital Networks -------------------------
        INC = IndustryNetworkCreation(
            year=year, input_filepath=input_filepath, output_filepath=output_filepath
        )
        INC.run()

        # Output
        data_path = os.path.join(output_filepath, year, "industry_output.parquet")
        INC.df_output.to_parquet(data_path)

        # GDP
        data_path = os.path.join(output_filepath, year, "gdp.parquet")
        INC.df_gdp.to_parquet(data_path)

        # Graph representation financial flows
        network_from_adjacency(adjacency_matrix=INC.A.T, # REMEMBER: io tables are transposed adj matrix
                               node_index=INC.node_index,                               
                               path = os.path.join(output_filepath, year, "A_country.graphml"),
                               tol_gfi=0.01,tol_favor=0.0001)
        
        # Graph representation goods and services flows
        network_from_adjacency(adjacency_matrix=INC.B, 
                               node_index=INC.node_index,
                               path = os.path.join(output_filepath, year, "B_country.graphml"),
                               tol_gfi=0.01,tol_favor=0.0001)
        
        # Migration Network --------------------------------------
        MNC = MigrationNetworkCreation(
            year=year, input_filepath=input_filepath, output_filepath=output_filepath
        )
        #MNC.run(source='un')
        MNC.run(source='oecd')

        # Compute network features
        NFC = NetworkFeatureComputation(MNC.G)
        NFC.compute_features(tol_gfi=0.00001, tol_favor=1e-15)

        # Save
        network_path = os.path.join(output_filepath, year, "migration_network.graphml")
        write_s3_graphml(NFC.G, network_path)
        
        # Estimated migration network ----------------------
        B = read_s3_graphml(os.path.join(output_filepath, year, "B_country.graphml"))
        emn = EstimatedMigrationNetwork(B, input_filepath, output_filepath)
        estimated_M = emn.estimate_emigration_rate()
        
        # Compute network features
        NFC = NetworkFeatureComputation(estimated_M)
        NFC.compute_features(tol_gfi=0.00001, tol_favor=0.001)

        # Save
        network_path = os.path.join(output_filepath, year, "estimated_migration_network.graphml")
        write_s3_graphml(NFC.G, network_path)
    
    etl = PanelDataETL(input_filepath=input_filepath, output_filepath=output_filepath)
    df_model = etl.run()

    df_model.to_parquet(os.path.join(output_filepath, "panel_data.parquet"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
