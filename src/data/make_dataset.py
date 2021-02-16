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


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

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
        # Generate networks -------------------------
        # Capital Network
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

        # Graph representation
        adj_matrix = INC.A.T  # REMEMBER: io tables are transposed adj matrix
        df_adj = pd.DataFrame(adj_matrix, index=INC.node_index, columns=INC.node_index)
        G = nx.convert_matrix.from_pandas_adjacency(df_adj, create_using=nx.DiGraph)
 
        # Subgaph with the countries under study 
        G = G.subgraph(countries_under_study)

        # Compute network features ------------------
        NFC = NetworkFeatureComputation(G)
        NFC.compute_features(tol_gfi=0.01, tol_favor=0.0001)
        G = NFC.G

        # Save
        network_path = os.path.join(output_filepath, year, "A_country.graphml")
        nx.readwrite.graphml.write_graphml(G, network_path)

        # Migration Network --------------------------------------
        MNC = MigrationNetworkCreation(
            year=year, input_filepath=input_filepath, output_filepath=output_filepath
        )
        MNC.run()

        # Subgaph with the countries under study 
        G = MNC.G.subgraph(countries_under_study)

        # Compute network features
        NFC = NetworkFeatureComputation(G)
        NFC.compute_features(tol_gfi=0.00001, tol_favor=1e-15)
        G = NFC.G

        # Save
        network_path = os.path.join(output_filepath, year, "migration_network.graphml")
        Path(network_path).parent.mkdir(parents=True, exist_ok=True)
        nx.readwrite.graphml.write_graphml(G, network_path)

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
