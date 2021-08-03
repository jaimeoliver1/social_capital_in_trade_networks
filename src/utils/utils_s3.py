import networkx as nx
from pathlib import Path
import os
import boto3
from urllib.parse import urlparse

def read_s3_graphml(path: str,
                    local_network_path = f'G.graphml'):
 
    o = urlparse(path, allow_fragments=False)
    bucket=o.netloc
    s3_path=o.path
    if s3_path[0] == '/': s3_path = s3_path[1:]

    s3 = boto3.resource('s3')
    s3.meta.client.download_file(bucket, s3_path, local_network_path)

    G = nx.readwrite.graphml.read_graphml(local_network_path)
 
    os.remove(local_network_path)
    
    return G

def write_s3_graphml(G,
                     path: str,
                     local_network_path = f'G.graphml'):
 
    o = urlparse(path, allow_fragments=False)
    bucket=o.netloc
    s3_path=o.path
    if s3_path[0] == '/': s3_path = s3_path[1:]

    nx.readwrite.graphml.write_graphml(G, local_network_path)

    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(local_network_path, bucket, s3_path)

    G = nx.readwrite.graphml.read_graphml(local_network_path)
 
    os.remove(local_network_path)
    
    return G