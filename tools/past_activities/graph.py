import torch


def edge2adj(edge, n_edge, n_vertex_max, device, add_self_ref=False):
    """
    Make adjacency matrix for each graph of batch.

    Parameters
    ----------
    edge: torch.tensor of torch.long (batch_size, n_edge_max, 2)
        List of indices of each edge.
        torch.long is recommended for type(edge) but other types are allowed.
    n_edge: array_like of int 
        List of number of edges in each graph.
    n_vertex_max: int
        Maximum number of vertex in one graph of batch.
    device: str
        Device which return tensor is stored in.
    add_self_ref: bool
        Whether or not to add self-referencing edge or not.
        If True, adj = adj + eye.

    Returns
    -------
    adj: torch.tensor of torch.float (batch_size, n_vertex_max, n_vertex_max)
        Adjacency matrix for each graph. Weight of each graph is currently
        unavailable.
    """
    edge = edge.to(device=device, dtype=torch.long)
    batch_size = len(edge)
    adj = torch.empty((batch_size, n_vertex_max, n_vertex_max),
        dtype=torch.float, device=device)
    for i in range(batch_size):
        e = edge[i,:n_edge[i]].transpose(0,1).to(device)
        adj[i] = torch.sparse.FloatTensor(e, 
            torch.ones(n_edge[i], device=device), torch.Size([n_vertex_max, n_vertex_max])).to_dense()
    
    adj = adj + adj.transpose(1, 2)
    if add_self_ref:
        adj = adj + torch.eye(n_vertex_max, dtype=torch.float, device=device).unsqueeze(0).expand(batch_size, n_vertex_max, n_vertex_max)
    return adj

    
def edge2lap(edge, n_edge, n_node_max, device, normalization='sym'):
    """
    Make laplacian matrix of graph from list of edge for each graph of batch.

    Parameters
    ----------
    edge: torch.tensor of torch.long (batch_size, n_edge_max, 2)
        List of indices of each edge.
        torch.long is recommended for type(edge) but other types are allowed.
    n_edge: array_like of int 
        List of number of edges in each graph.
    n_node_max: int
        Maximum number of vertex in one graph of batch.
    device: str
        Device which return tensor is stored in.
    normalization: str, 'none', 'sym' or 'rw'
        Type of normalization for laplacian matrix.
        See reference of torch_geometric.nn.ChebComb
    
    Returns
    -------
    lap: torch.tensor of torch.float (batch_size, n_vertex_max, n_vertex_max)
        Laplacian matrix of each graph in batch.
    """
    if normalization not in ['none', 'sym', 'rw']:
        raise ValueError("'normalization' must be either 'none', 'sym', or 'rw'")

    adj = edge2adj(edge, n_edge, n_node_max, device)
    dims = adj.sum(dim=2)
    node_dim_zero = (dims == 0).to(torch.float)
    dims += node_dim_zero

    lap = torch.diag_embed(dims, dim1=1, dim2=2) - adj
    if normalization == 'sym':
        dims = torch.diag_embed(dims.pow(-0.5), dim1=1, dim2=2)
        lap = torch.bmm(dims, torch.bmm(lap, dims))
    elif normalization == 'rw':
        dims = torch.diag_embed(dims.pow(-1), dim1=1, dim2=2)
        lap = torch.bmm(dims, lap)
    lap -= torch.diag_embed(node_dim_zero, dim1=1, dim2=2)
    return lap

def graph_max_pool(nodes, adj, add_self_ref=True):
    """
    This function substitutes vector of each node with the max value in vectors of the jointed nodes including the node itself.

    Parameters
    ----------
    nodes: torch.tensor of any type (batch_size, n_node_max, in_features)
        Feature vectors of each node.
    adj: torch.tensor of torch.float (batch_size, n_node_max, n_node_max)
        Adjacency matrix of each sample. Weights of edges is currently not considered.
        Each value must be 0 or 1.
    add_self_ref: bool
        Whether to add reference to self or not

    Returns
    -------
    x: torch.tensor of any type (batch_size, n_node_max, in_features)
    """
    batch_size, n_node_max, in_features = nodes.shape
    if add_self_ref:
        adj_t = adj + torch.eye(n_node_max).unsqueeze(0).expand(batch_size, n_node_max, n_node_max)
    else:
        adj_t = adj
    
    return torch.max(adj_t.unsqueeze(3).expand(*adj.shape, in_features)*\
        nodes.unsqueeze(1).expand(batch_size, n_node_max, n_node_max, in_features),
        axis=2).values
