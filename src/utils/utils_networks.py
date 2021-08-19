import networkx as nx
import numpy as np
import os

def favor_centrality(G, tol=0.0001):

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('cannot compute centrality for the null graph')
        
    g = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    g2 = g.dot(g)

    supported_friends = (g2 > tol) & (g > tol)

    favor_centrality_list =  supported_friends.sum(axis=1)

    favor_centrality_list = g2.sum(axis=1)

    return dict(zip(G, favor_centrality_list)) 

def bridging_centrality(G, p=1, T=5):

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('cannot compute centrality for the null graph')
        
    g = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()

    p_matrix = p*g

    bridging_centrality_list = []

    for i in range(len(g)):
        i_total = 0

        for j in range(len(g)):
            p_matrix_ij = p_matrix.copy()
            p_matrix_ij[i,j] = 0

            i_total += np.sum([np.linalg.matrix_power(p_matrix, t) - np.linalg.matrix_power(p_matrix_ij, t) for t in range(1,T+1)])

        bridging_centrality_list.append(i_total)


    return dict(zip(G, bridging_centrality_list))

    
def godfhater_index(G, tol=1.0e-10):

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('cannot compute centrality for the null graph')
        
    
    g = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    
    godfhater_index_list = []
    for index, _ in enumerate(G.nodes):
        prod = np.tensordot(g[:,index],g[:,index], axes = 0)
        prod_k_bigger_than_j = np.tril(prod,k=-1)
        
        prod_k_bigger_than_j = prod_k_bigger_than_j[(g.T < tol) & (g < tol)]

        gf_i = prod_k_bigger_than_j.sum()

        godfhater_index_list.append(gf_i)


    return dict(zip(G, godfhater_index_list))


def average_degree(G, weight='weight'):
    return sum(dict(G.degree(weight='weight')).values())/float(len(G))

def global_efficiency(G, weight='weight'):
    """Returns the average global efficiency of the graph.

    The *efficiency* of a pair of nodes in a graph is the multiplicative
    inverse of the shortest path distance between the nodes. The *average
    global efficiency* of a graph is the average efficiency of all pairs of
    nodes [1]_.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        An undirected graph for which to compute the average global efficiency.

    Returns
    -------
    float
        The average global efficiency of the graph.

    Notes
    -----
    Edge weights are ignored when computing the shortest path distances.

    See also
    --------
    local_efficiency

    References
    ----------
    .. [1] Latora, Vito, and Massimo Marchiori.
           "Efficient behavior of small-world networks."
           *Physical Review Letters* 87.19 (2001): 198701.
           <https://doi.org/10.1103/PhysRevLett.87.198701>
    """
    if nx.is_negatively_weighted(G, weight=weight):
        raise nx.NetworkXError("edge weights must be positive")
        
    total_weight = G.size(weight=weight)
    if total_weight <= 0:
        raise nx.NetworkXError("Size of G must be positive")
    
    denom = len(G.edges)
    
    if denom != 0:

        def as_distance(u, v, d):
            distance = d.get(weight, 1)
            return total_weight / distance if distance>0 else np.inf
        lengths = nx.all_pairs_dijkstra_path_length(G, weight=as_distance)
        
        g_eff = 0
        for source, targets in lengths:
            for target, distance in targets.items():
                if distance > 0:
                    g_eff += 1 / distance
        g_eff /= denom
        # g_eff = sum(1 / d for s, tgts in lengths
        #                   for t, d in tgts.items() if d > 0) / denom
    else:
        g_eff = 0
        
    return g_eff

def network_years_generator(output_filepath, network):
    '''
    Generator of the sequence of networks over the years
    '''
    all_years = []
    for y in range(2000, 2019):
        network_path = os.path.join(output_filepath, str(y), f'{network}.graphml')
        G = nx.readwrite.graphml.read_graphml(network_path)
        
        all_years.append(G)
        
    return all_years