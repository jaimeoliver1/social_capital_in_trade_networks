# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from utils.utils_features import NetworkFeatureComputation
from data.financial_network import IndustryNetworkCreation
from data.migration_network import MigrationNetworkCreation
from data.etl import ETL

import pandas as pd
import networkx as nx

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    for year in range(2005, 2016):
        print('Processing year ', str(year))

        # Generate networks -------------------------
        # Capital Network
        INC = IndustryNetworkCreation(year=str(year))
        INC.oecd_matrix_ingestion(by_country=True)
        INC.upstream_chain()
        INC.downstream_chain()
        
        df_out = pd.DataFrame(INC.x, index=INC.node_index, columns=['OUTPUT'])
        df_out.to_parquet(os.path.join(output_filepath, year, 'industry_outputs.parquet'))

        # GDP
        df_gdp = pd.DataFrame(INC.w, index=INC.node_index, columns=['gdp'])
        df_gdp.to_parquet(os.path.join(output_filepath, year,"gdp.parquet"))

        # Save transition matrix
        adj_matrix = INC.A.T # REMEMBER: io tables are transposed adj matrix  
        df_adj = pd.DataFrame(adj_matrix, index=INC.node_index, columns=INC.node_index)
        G = nx.convert_matrix.from_pandas_adjacency(df_adj, create_using=nx.DiGraph)

        # Compute network features ------------------
        NFC = NetworkFeatureComputation(G)
        NFC.compute_features(tol_favor=0.001)
        G = NFC.G

        # Save
        network_path = os.path.join(output_filepath, year,'A_country.graphml')
        Path(network_path).parent.mkdir(parents=True, exist_ok=True)
        nx.readwrite.graphml.write_graphml(G, network_path)
            
        # Migration Network --------------------------------------
        MNC = MigrationNetworkCreation(year=year)
        MNC.run()

        # Compute network features 
        NFC = NetworkFeatureComputation(MNC.G)
        NFC.compute_features(tol_favor=0)
        G = NFC.G    

        # Save
        network_path = os.path.join(output_filepath,year, 'migration_network.graphml')
        Path(network_path).parent.mkdir(parents=True, exist_ok=True)
        nx.readwrite.graphml.write_graphml(G, network_path)

    etl = ETL(input_filepath = input_filepath, output_filepath = output_filepath)
    df_model = etl.run()

    df_model.to_parquet(os.path.join(output_filepath,'panel_data.parquet'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
