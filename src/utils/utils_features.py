import pandas as pd
import numpy as np
import networkx as nx
from src.utils.utils_networks import godfhater_index, bridging_centrality, favor_centrality

import os
from pathlib import Path
from collections import defaultdict


class NetworkFeatureComputation:
    def __init__(self, graph):
        self.G = graph
        
    def compute_features(self, tol_gfi, tol_favor):
        '''
        Compute graph features
        '''
        self.df = pd.DataFrame(list(self.G.nodes), columns=['country_industry'])
        
        pagerank_dict = nx.pagerank(self.G)
        pagerank_dict = {k:{'pagerank':v} for k,v in pagerank_dict.items()}
        nx.set_node_attributes(self.G, pagerank_dict)
        
        gfi = godfhater_index(self.G, tol=tol_gfi)
        gfi = {k:{'gfi':v} for k,v in gfi.items()}
        nx.set_node_attributes(self.G, gfi)     
        
        bridging = bridging_centrality(self.G)
        bridging = {k:{'bridging':v} for k,v in bridging.items()}
        nx.set_node_attributes(self.G, bridging)     
        
        favor = favor_centrality(self.G, tol=tol_favor)
        favor = {k:{'favor':v} for k,v in favor.items()}
        nx.set_node_attributes(self.G, favor)   
        
        g = nx.linalg.graphmatrix.adjacency_matrix(self.G).toarray()
        g = np.nan_to_num(g)
        
        hhi_index = np.square(g/g.sum(axis=1)[:,None]).sum(axis=1)   
        hhi_index = dict(zip(self.G.nodes(), hhi_index))
        hhi_index = {k:{'hhi_index':hhi_index_i} for k, hhi_index_i in hhi_index.items()}
        nx.set_node_attributes(self.G, hhi_index)
