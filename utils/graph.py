import numpy as np 
import networkx as nx 
from sklearn.neighbors import KDTree

def make_graph(point_cloud, n=9):
    """Create a k-nearest neighbor graph from the point cloud."""
    tree = KDTree(point_cloud)
    g = nx.Graph()
    g.add_nodes_from(list(range(len(point_cloud))))
    edge_list = []
    for k in range(len(point_cloud)):
        point = point_cloud[k]
        indices = tree.query([point], k=n+1, return_distance=False)
        for i in indices[0][1:]:
            g.add_edge(k,i)
    return g.to_undirected()    

def write_graph(graph: nx.Graph, labels, path):
    """Dump the graph data to be postprocessesed (smoothed) using graph cut.
    
    Parameters
    ----------
    graph
        KNN graph.
    labels
        Labels of the graph nodes.
    path
        Common path for both files.
    """
    np.savetxt(path+'edges.txt', graph.edges(),fmt='%i')
    # save nodes file as list of integer labels (= unary potentials)
    np.savetxt(path+'nodes.txt', np.asarray(labels).astype(int), fmt='%i')
