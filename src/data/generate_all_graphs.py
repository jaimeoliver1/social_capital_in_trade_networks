from utils import NetworkFeatureComputation
from financial_network import IndustryNetworkCreation
from migr import MigrationNetworkCreation

import pandas as pd
import networkx as nx

from pathlib import Path

save_to = '/domino/datasets/jaime_oliver/industry_network/scratch/'

for year in range(2005, 2016):
    print('Processing year ', str(year))

    # Generate networks -------------------------
    # Capital Network
    INC = IndustryNetworkCreation(year=str(year))
    INC.oecd_matrix_ingestion(by_country=True)
    INC.upstream_chain()
    INC.downstream_chain()
    
    df_out = pd.DataFrame(INC.x, index=INC.node_index, columns=['OUTPUT'])
    df_out.to_parquet(f'{save_to}{year}/industry_outputs.parquet')

    # GDP
    df_gdp = pd.DataFrame(INC.w, index=INC.node_index, columns=['gdp'])
    df_gdp.to_parquet(f"{save_to}{year}/gdp.parquet")

    for net in ['A', 'B', 'Z']:
        # Save transition matrix
        adj_matrix = getattr(INC, net).T # REMEMBER: io tables are transposed adj matrix  
        df_adj = pd.DataFrame(adj_matrix, index=INC.node_index, columns=INC.node_index)
        G = nx.convert_matrix.from_pandas_adjacency(df_adj, create_using=nx.DiGraph)

        # Compute network features ------------------
        NFC = NetworkFeatureComputation(G)
        NFC.compute_features(tol_favor=0.001)
        G = NFC.G

        # Save
        network_path = f'{save_to}{year}/{net}_country.graphml'
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
    network_path = f'{save_to}{year}/migration_network.graphml'
    Path(network_path).parent.mkdir(parents=True, exist_ok=True)
    nx.readwrite.graphml.write_graphml(G, network_path)
    
    