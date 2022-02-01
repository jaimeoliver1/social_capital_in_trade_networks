import pandas as pd
import numpy as np
import networkx as nx
from src.utils.utils_networks import godfhater_index, bridging_centrality, favor_centrality

import os
from pathlib import Path
from collections import defaultdict
import warnings

class NetworkFeatureComputation:
    def __init__(self, graph):
        self.G = graph
        
    def compute_features(self, tol_gfi, tol_favor):
        '''
        Compute graph features
        '''
        self.df = pd.DataFrame(list(self.G.nodes), columns=['country_industry'])
        
        try:
            h,a=nx.hits(self.G, max_iter=750)
            h = {k:{'hubs':v} for k,v in h.items()}
            a = {k:{'authorities':v} for k,v in a.items()}

        except nx.PowerIterationFailedConvergence:
            warnings.warn("nx.PowerIterationFailedConvergence")
            h = {n:{'hubs':np.nan} for n in self.G.nodes()}
            a = {n:{'authorities':np.nan} for n in self.G.nodes()}

        nx.set_node_attributes(self.G, h)
        nx.set_node_attributes(self.G, a)

        try:
            pagerank_dict = nx.pagerank(self.G, max_iter=1000, weight='weight')
            pagerank_dict = {k:{'pagerank':v} for k,v in pagerank_dict.items()}

        except nx.PowerIterationFailedConvergence:
            warnings.warn("nx.PowerIterationFailedConvergence")
            pagerank_dict = {n:{'pagerank':np.nan} for n in self.G.nodes()}

        nx.set_node_attributes(self.G, pagerank_dict)
        
        gfi = godfhater_index(self.G, tol=tol_gfi)
        gfi = {k:{'gfi':v} for k,v in gfi.items()}
        nx.set_node_attributes(self.G, gfi)     
        
        bridging = bridging_centrality(self.G)
        bridging = {k:{'bridging':v} for k,v in bridging.items()}
        nx.set_node_attributes(self.G, bridging)     
        
        out_favor = favor_centrality(self.G, tol=tol_favor)
        out_favor = {k:{'out_favor':v} for k,v in out_favor.items()}
        nx.set_node_attributes(self.G, out_favor)   

        in_favor = favor_centrality(self.G, tol=tol_favor, transpose=True)
        in_favor = {k:{'in_favor':v} for k,v in in_favor.items()}
        nx.set_node_attributes(self.G, in_favor)   
        
        g = nx.linalg.graphmatrix.adjacency_matrix(self.G).toarray()
        g = np.nan_to_num(g)
        
        hhi_index = np.square(g/g.sum(axis=1)[:,None]).sum(axis=1)   
        hhi_index = dict(zip(self.G.nodes(), hhi_index))
        hhi_index = {k:{'hhi_index':hhi_index_i} for k, hhi_index_i in hhi_index.items()}
        nx.set_node_attributes(self.G, hhi_index)
