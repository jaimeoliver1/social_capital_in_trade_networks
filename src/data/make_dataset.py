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
from src.data.migration_network import MigrationNetworkCreation
from src.data.panel_data_etl import PanelDataETL

from src.utils.utils_s3 import read_s3_graphml, write_s3_graphml


def network_from_adjacency(adjacency_matrix, 
                           node_index, 
                           bucket,
                           network_path, 
                           tol_gfi=0.01, 
                           tol_favor=0.0001):
        df_adj = pd.DataFrame(adjacency_matrix, index=node_index, columns=node_index)
        G = nx.convert_matrix.from_pandas_adjacency(df_adj, create_using=nx.DiGraph)
 
        # Compute network features ------------------
        NFC = NetworkFeatureComputation(G)
        NFC.compute_features(tol_gfi=tol_gfi, tol_favor=tol_favor)
        G = NFC.G

        # Save
        write_s3_graphml(G, bucket, network_path)

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    bucket='workspaces-clarity-mgmt-pro'
    s3_path='jaime.oliver/misc/social_capital/data/'

    for year in range(2000, 2019):

        year = str(year)
        print("Processing year ", year)

        countries_under_study = [
            "AUS",
            "AUT",
            "BEL",
            "CAN",
            "CZE",
            "DNK",
            "FIN",
            "FRA",
            "DEU",
            "HUN",
            "ISL",
            "ITA",
            "JPN",
            "KOR",
            "LUX",
            "NLD",
            "NZL",
            "NOR",
            "POL",
            "SVK",
            "ESP",
            "SWE",
            "CHE",
            "GBR",
            "USA",
            "CHL",
            "EST",
            "GRC",
            "MEX",
            "SVN",
            "PRT",
            "ISR",
            "IRL",
            "LVA",
        ]
        
        # Capital Networks -------------------------
        INC = IndustryNetworkCreationEORA(
            year=year, input_filepath=input_filepath, output_filepath=output_filepath
        )
        INC.run()

        # Output
        data_path = os.path.join(output_filepath, year, "industry_output.parquet")
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        INC.df_output.to_parquet(data_path)

        # GDP
        data_path = os.path.join(output_filepath, year, "gdp.parquet")
        INC.df_gdp.to_parquet(data_path)

        
        # Graph representation financial flows
        network_from_adjacency(adjacency_matrix=INC.A.T, # REMEMBER: io tables are transposed adj matrix
                               node_index=INC.node_index,                               
                               bucket = bucket,
                               network_path = os.path.join(s3_path, year, "A_country.graphml"),
                               tol_gfi=0.01,tol_favor=0.0001)
        
        # Graph representation goods and services flows
        network_from_adjacency(adjacency_matrix=INC.B.T, # REMEMBER: io tables are transposed adj matrix
                               node_index=INC.node_index,
                               bucket = bucket,
                               network_path = os.path.join(s3_path, year, "B_country.graphml"),
                               tol_gfi=0.01,tol_favor=0.0001)

        # Migration Network --------------------------------------
        
        MNC = MigrationNetworkCreation(
            year=year, input_filepath=input_filepath, output_filepath=output_filepath
        )
        MNC.run()

        # Subgaph with the countries under study 
        #G = MNC.G.subgraph(countries_under_study)

        # Compute network features
        NFC = NetworkFeatureComputation(MNC.G)
        NFC.compute_features(tol_gfi=0.00001, tol_favor=1e-15)
        G = NFC.G

        # Save
        network_path = os.path.join(s3_path, year, "migration_network.graphml")
        write_s3_graphml(G, bucket, network_path)
        
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
