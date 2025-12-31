# K-Path Cover
import random

import networkx as nx
import numpy as np

from tqdm import tqdm
import copy
from .pq import PriorityQueue

def r_kpathcover(g, k, directed=False):
    """Compute a k-path cover based on the random strategy.

        # Arguments
            g: a networkx graph or dgl.Graph
            k: the integer value k
            directed: bool, directed vs. undirected
    """
    
    # Sort nodes in random order
    nodes = sorted(g.degree, key=lambda x: random.random(), reverse=True)

    C = set()
    # For each node v, determine whether v is pruned or not.
    for v, _ in nodes:
        flag, ret = path_search(g, v, C, k, directed)
        if flag:
            C.add(v)
            continue

        for path in ret:
            flag, _ = path_search(g, v, C, k, directed, path, inbound=True)
            if flag:
                C.add(v)
                break
    return C


def p_kpathcover(g, k, directed=False, ordered_nodes=None):
    """Compute a k-path cover based on the pruning strategy.

        # Arguments
            g: a networkx graph or dgl.Graph
            k: the integer value k
            directed: bool, directed vs. undirected
    """
    
    # Sort nodes in some order
    if ordered_nodes is None:
        nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)
    else:
        nodes = ordered_nodes

    C = set()
    # For each node v, determine whether v is pruned or not.
    for v, _ in nodes:

        flag, ret = path_search(g, v, C, k, directed)
        if flag:
            C.add(v)
            continue

        for path in ret:
            flag, _ = path_search(g, v, C, k, directed, path, inbound=True)
            if flag:
                C.add(v)
                break
    return C

def i_kpathcover(g, k, directed=False):

    if not g.is_directed():
        g = g.to_directed()
    for _ in range(k):
        g = isset_overlay(g)
    C = [ n for n in g.nodes ]
    return C

def path_search(g, v, C, k, directed, input_path=None, inbound=False):
    """
        # Arguments
            g: a networkx graph
            v: a vertex id
            C: a k-path cover set
            k: the integer value k
            directed: bool, directed vs. undirected
            input_path: an intermediate path to v
            inbound: bool, inbound search vs. outbound search
    """
    if k == 0:
        return True, None # No need to test.

    # search initialization
    if input_path is not None:
        assert inbound
        stk = [(v, input_path)]
        ret = None
    else:
        stk = [(v, {v})]
        ret = []

    # search
    while len(stk) > 0:
        curr, path = stk.pop()
        if ret is not None:
            ret.append(path)

        if directed:
            if inbound:
                edges = g.in_edges
            else:
                edges = g.out_edges
        else:
            edges = g.edges

        for s, t in list(edges(curr)):
            if directed and inbound:
                next_ = s
            else:
                next_ = t

            if next_ in C: # Avoid to visit a cover node
                continue
            if input_path is not None and next_ in input_path:
                continue
                
            # expanding
            if next_ not in path:
                path_ = copy.deepcopy(path)
                path_.add(next_)

                if not inbound: # outbound path search
                    if len(path_) > k:
                        return True, ret
                else:
                    if len(path_) - 1 > k: # input_path contains v.
                        return True, ret
                stk.append((next_, path_))
    return False, ret

def overlay(g, C, directed=False, tqdm_disable=False):
    if directed:
        d = nx.DiGraph()
    else:
        d = nx.Graph()

    for c in C:
        d.add_node(c)

    for c in tqdm(C, disable=tqdm_disable):
        reach = bounded_traverse(g, c, C, directed)
        for r in reach:
            assert c != r
            d.add_edge(c, r)
    return d

def bounded_traverse(g, v, C, directed=False, reversely=False, ret_visit=False):
    visit = {v}
    stk = [v]
    reach = set()
    while len(stk) > 0:
        curr = stk.pop()
        if directed:
            if reversely:
                edges = g.in_edges
            else:
                edges = g.out_edges
        else:
            edges = g.edges

        for _, next_ in list(edges(curr)):
            if next_ in C and next_ != v and next_ not in visit:
                reach.add(next_)
                visit.add(next_)
                continue
            if next_ not in visit:
                stk.append(next_)
                visit.add(next_)
    if ret_visit:
        return visit
    return reach

def dgl_overlay(g, k, device, add_self_loop=False, add_local_edge=False, reversely=False, residual=False, exclude_overlay=False):
    nxg = g.to_networkx()
    C = p_kpathcover(nxg, k)
    if exclude_overlay:
        d = nx.DiGraph()
    else:
        d = overlay(nxg, C)
    if residual:
        for s, t, _ in nxg.edges:
            if not d.has_edge(s,t):
                d.add_edge(s,t)

    d_in = { n:-1 for n in d.nodes }
    for n in nxg.nodes:
        if n not in d_in:
            d.add_node(n)

    if add_local_edge:
        for v in C:
            visit = bounded_traverse(nxg, v, C, directed=True, reversely=reversely, ret_visit=True)
            for u in visit:
                d.add_edge(v, u)

    d = dgl.from_networkx(d, device=device)
    if add_self_loop:
        d = dgl.remove_self_loop(d)
        d = dgl.add_self_loop(d)
    return d


def isset_overlay(g):
    g = copy.deepcopy(g)
    pq = PriorityQueue()
    for n in g.nodes:
        cost = 0
        for p in g.predecessors(n):
            for s in g.successors(n):
                if not g.has_edge(p, s):
                    cost += 1
        es = cost - (g.out_degree(n) + g.in_degree(n))
        pq.add_task(n, priority=es)

    while pq.get_len() > 0:
        n, priority = pq.pop_task()

        # n is removed from Gi
        predecessors = [p for p in g.predecessors(n)]
        successors = [s for s in g.successors(n)]
        for p in predecessors:
            if pq.is_included(p):
                pq.remove_task(p)
        for s in successors:
            if pq.is_included(s):
                pq.remove_task(s)

        g.remove_node(n)
        for p in predecessors:
            for s in successors:
                if not g.has_edge(p, s) and p != s and p != n and s != n:
                    preds = set(g.predecessors(s))
                    for s_ in g.successors(p):
                        if s_ in preds and pq.is_included(s_):
                            es = pq.get_priority(s_)
                            es -= 1
                            pq.add_task(s_, priority=es)
                    g.add_edge(p, s)
    return g 

# independent set-based
def dgl_overlay2(g, k, device, add_self_loop=False, add_local_edge=False, reversely=False, residual=False, exclude_overlay=False):
    import dgl
    nxg = g.to_networkx()
    d = nxg
    for _ in range(k):
        d = isset_overlay(d)
    C = [ n for n in d.nodes ]
    if exclude_overlay:
        d = nx.DiGraph()

    if residual:
        for s, t, _ in nxg.edges:
            if not d.has_edge(s,t):
                d.add_edge(s,t)

    d_in = { n:-1 for n in d.nodes }
    for n in nxg.nodes:
        if n not in d_in:
            d.add_node(n)

    if add_local_edge:
        for v in C:
            visit = bounded_traverse(nxg, v, C, directed=True, reversely=reversely, ret_visit=True)
            for u in visit:
                if not d.has_edge(v,u):
                    d.add_edge(v, u)

    d = dgl.from_networkx(d, device=device)
    if add_self_loop:
        d = dgl.remove_self_loop(d)
        d = dgl.add_self_loop(d)
    return d


def adj_pg_overlay(adj, k, add_local_edge=False, reversely=False, residual=False, exclude_overlay=False):
    nxg = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)
    """
    d = nxg
    for _ in range(k):
        d = isset_overlay(d)
    C = [ n for n in d.nodes ]
    """

    pr = nx.pagerank(nxg)
    prlist = []
    for name, value in pr.items():
        prlist.append((name, value))
    prlist = sorted(prlist, key=lambda x: x[1])
    prlist.reverse()
    
    C = []
    for _ in range(k):
        C.append(prlist[_][0])

    d = overlay(nxg, C, directed=True)
    d = nx.compose(d, nxg)
    
    """
    d_in = { n:-1 for n in d.nodes }
    for n in nxg.nodes:
        if n not in d_in:
            d.add_node(n)

    if residual:
        for s, t in nxg.edges:
            if not d.has_edge(s,t):
                d.add_edge(s,t)
    """

    if add_local_edge:
        for v in C:
            visit = bounded_traverse(nxg, v, C, directed=True, reversely=reversely, ret_visit=True)
            for u in visit:
                if u != v:
                    if not d.has_edge(v, u):
                        d.add_edge(v, u)

    #d.remove_edges_from(nx.selfloop_edges(d))
    """
    for n in d.nodes: # self-edge
        if not d.has_edge(n, n):
            d.add_edge(n, n)
    """

    new_adj = nx.to_scipy_sparse_array(d, nodelist=sorted(d.nodes()))
    return new_adj

def nx_overlay(nxg, k, add_local_edge=False, reversely=False, residual=False, exclude_overlay=False, hit=False, features=None):

    d = nxg
    temp = []
    for _ in range(k):

        #pr = nx.pagerank(d)
        #pr = nx.global_reaching_centrality(d)
        #pr = nx.eigenvector_centrality(d)
        #pr = nx.laplacian_centrality(d)
        #pr = nx.percolation_centrality(d)
        #pr = nx.subgraph_centrality(d)

        if hit:
            #pr = nx.closeness_centrality(d)
            pr = nx.pagerank(d)
        else:
            #pr = { n: random.random() for n in d.nodes }
            pr = nx.eigenvector_centrality(d)

        prlist = []
        for name, value in pr.items():
            prlist.append((name, value))
        prlist = sorted(prlist, key=lambda x: x[1])
        prlist.reverse()

        C = p_kpathcover(d, 1, directed=True, ordered_nodes=prlist)
        d = overlay(d, C, directed=True)
        """
        removed = []
        for (u,v) in d.edges:
            if not is_sim(u,v):
                removed.append((u,v))
        print(len(removed), " edges are removed.")
        d.remove_edges_from(removed)
        """
        temp.append((d, C))

    d = nxg
    C = []
    for d_, C_ in temp:
        d = nx.compose(d, d_)
        C.extend(C_)
        C = list(set(C))
    C = sorted(C)

    if add_local_edge:
        for v in C:
            visit = bounded_traverse(nxg, v, C, directed=True, reversely=reversely, ret_visit=True)
            for u in visit:
                if u != v:
                    if not d.has_edge(v, u) and is_sim(v, u):
                        d.add_edge(v, u)

    return d


def adj_overlay(adj, k, add_local_edge=False, reversely=False, features=None, nodelist=None, method="pagerank", ret_mode="acc", min_cover_nodes=0, tqdm_disable=False):

    if nodelist is not None:
        import copy
        if type(nodelist[0]) == int:
            nodelist = copy.deepcopy(nodelist)
            for i in range(len(nodelist)):
                nodelist[i] = (nodelist[i], -1) # int to tuple

    def is_sim(u, v):
        vec1 = features[u].toarray()[0]
        vec2 = features[v].toarray()[0]
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        sim = dot_product / (norm_vec1 * norm_vec2)
        return sim >= 0.06

    nxg = nx.from_scipy_sparse_array(adj, create_using=nx.Graph)

    d = nxg
    temp = []
    for _ in range(k):

        if nodelist is None:
            #pr = nx.global_reaching_centrality(d)
            #pr = nx.eigenvector_centrality(d)
            #pr = nx.laplacian_centrality(d)
            #pr = nx.percolation_centrality(d)
            #pr = nx.subgraph_centrality(d)

            if "pagerank" in method:
                pr = nx.pagerank(d)
            elif "closeness_centrality" in method:
                pr = nx.closeness_centrality(d)
            elif "random" in method:
                pr = { n: random.random() for n in d.nodes }
            elif "eigenvector_centrality" in method:
                pr = nx.eigenvector_centrality(d)
            else:
                raise NotImplementedError()

            nodelist = []
            for name, value in pr.items():
                nodelist.append((name, value))
            nodelist = sorted(nodelist, key=lambda x: x[1])
            if "reverse" in method:
                nodelist.reverse()

        else: # conduct filtering

            nodelist_ = []
            for v, _ in nodelist:
                if v in d.nodes:
                    nodelist_.append((v, _))
            nodelist = nodelist_

        C = p_kpathcover(d, 1, directed=False, ordered_nodes=nodelist)

        if min_cover_nodes > 0 and len(C) < min_cover_nodes and len(temp) > 0:
            break
            
        d = overlay(d, C, directed=False, tqdm_disable=tqdm_disable)

        if features:
            removed = []
            for (u,v) in d.edges:
                if not is_sim(u,v):
                    removed.append((u,v))
            d.remove_edges_from(removed)

        temp.append((d, C))

    if ret_mode == "overlay":
        new_adj = nx.to_scipy_sparse_array(temp[-1][0], nodelist=sorted(temp[-1][0].nodes()))

        # local edge
        d = nx.Graph()
        d.add_nodes_from(nxg.nodes(data=True))

        for v in tqdm(C, disable=tqdm_disable):
            visit = bounded_traverse(nxg, v, C, directed=False, reversely=reversely, ret_visit=True)
            for u in visit:
                if u != v:
                    #if not d.has_edge(v, u) and is_sim(v, u):
                    if not d.has_edge(v, u):
                        d.add_edge(v, u)

        return new_adj, d

    elif ret_mode == "acc":
        d = nxg
        C = []
        for d_, C_ in temp:
            d = nx.compose(d, d_)
            C.extend(C_)
            C = list(set(C))
        C = sorted(C)

    elif ret_mode == "local":
        d = nx.Graph()
        d.add_nodes_from(nxg.nodes(data=True))
        C = temp[-1][1]
        add_local_edge = True

    elif ret_mode == "overlay_local":
        #new_adj = nx.to_scipy_sparse_array(temp[-1][0], nodelist=sorted(temp[-1][0].nodes()))
        # local edge
        d = nx.Graph()
        d.add_nodes_from(nxg.nodes(data=True))

        for u, v in temp[-1][0].edges:
            d.add_edge(u,v)
        C = temp[-1][1]
        add_local_edge = True

    elif ret_mode == "last": 
        d = nxg
        d = nx.compose(d, temp[-1][0])
        C = temp[-1][1]
    else:
        raise NotImplementedError(ret_mode + " not implemented!")

    if add_local_edge:
        for v in tqdm(C, disable=tqdm_disable):
            visit = bounded_traverse(nxg, v, C, directed=False, reversely=reversely, ret_visit=True)
            for u in visit:
                if u != v:
                    #if not d.has_edge(v, u) and is_sim(v, u):
                    if not d.has_edge(v, u):
                        d.add_edge(v, u)

    new_adj = nx.to_scipy_sparse_array(d, nodelist=sorted(d.nodes()))
    return new_adj



def adj_overlay_backup(adj, k, add_local_edge=False, reversely=False, hit=False, features=None, prlist=None):

    if prlist is not None:
        import copy
        if type(prlist[0]) == int:
            prlist = copy.deepcopy(prlist)
            for i in range(len(prlist)):
                prlist[i] = (prlist[i], -1) # int to tuple

    def is_sim(u, v):
        vec1 = features[u].toarray()[0]
        vec2 = features[v].toarray()[0]
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        sim = dot_product / (norm_vec1 * norm_vec2)
        return sim >= 0.06

    nxg = nx.from_scipy_sparse_array(adj, create_using=nx.Graph)

    d = nxg
    temp = []
    for _ in range(k):

        if prlist is None:
            #pr = nx.pagerank(d)
            #pr = nx.global_reaching_centrality(d)
            #pr = nx.eigenvector_centrality(d)
            #pr = nx.laplacian_centrality(d)
            #pr = nx.percolation_centrality(d)
            #pr = nx.subgraph_centrality(d)

            if hit:
                #pr = nx.closeness_centrality(d)
                pr = nx.pagerank(d)
            else:
                #pr = { n: random.random() for n in d.nodes }
                pr = nx.eigenvector_centrality(d)

            prlist = []
            for name, value in pr.items():
                prlist.append((name, value))
            prlist = sorted(prlist, key=lambda x: x[1])
            #prlist.reverse()

        else: # conduct filtering

            prlist_ = []
            for v, _ in prlist:
                if v in d.nodes:
                    prlist_.append((v, _))
            prlist = prlist_

        C = p_kpathcover(d, 1, directed=False, ordered_nodes=prlist)
        d = overlay(d, C, directed=False)
        """
        removed = []
        for (u,v) in d.edges:
            if not is_sim(u,v):
                removed.append((u,v))
        print(len(removed), " edges are removed.")
        d.remove_edges_from(removed)
        """
        temp.append((d, C))

    d = nxg
    C = []
    for d_, C_ in temp:
        d = nx.compose(d, d_)
        C.extend(C_)
        C = list(set(C))
    C = sorted(C)

    if add_local_edge:
        for v in C:
            visit = bounded_traverse(nxg, v, C, directed=False, reversely=reversely, ret_visit=True)
            for u in visit:
                if u != v:
                    #if not d.has_edge(v, u) and is_sim(v, u):
                    if not d.has_edge(v, u):
                        d.add_edge(v, u)
    print(d, k)
    new_adj = nx.to_scipy_sparse_array(d, nodelist=sorted(d.nodes()))
    return new_adj


def adj_overlay_(adj, k, add_local_edge=False, reversely=False, residual=False, exclude_overlay=False):
    nxg = nx.from_scipy_sparse_array(adj, create_using=nx.Graph)

    """
    temp = []
    d = nxg
    for _ in range(k):
        d = isset_overlay(d)
        temp.append(d)
    C = [ n for n in d.nodes ]

    d = nxg    
    for d_ in temp:
        d = nx.compose(d, d_)
    """

    d = nxg
    print(d)
    temp = []
    for _ in range(k):

        #pr = nx.pagerank(d)
        #pr = nx.closeness_centrality(d)
        #pr = nx.global_reaching_centrality(d)
        pr = nx.eigenvector_centrality(d)
        #pr = nx.laplacian_centrality(d)
        #pr = nx.percolation_centrality(d)
        #pr = nx.communicability_betweenness_centrality(d)

        prlist = []
        for name, value in pr.items():
            prlist.append((name, value))
        prlist = sorted(prlist, key=lambda x: x[1])
        prlist.reverse()

        C = p_kpathcover(d, 1, directed=False, ordered_nodes=prlist)
        #C = p_kpathcover(d, 1, directed=True, ordered_nodes=None)
        d = overlay(d, C, directed=False)
        temp.append((d, C))

    d = nxg
    C = []
    for d_, C_ in temp:
        d = nx.compose(d, d_)
        C.extend(C_)
        C = list(set(C))
    C = sorted(C)

    if add_local_edge:
        for v in C:
            visit = bounded_traverse(nxg, v, C, directed=False, reversely=reversely, ret_visit=True)
            for u in visit:
                if u != v:
                    if not d.has_edge(v, u):
                        d.add_edge(v, u)

    #d.remove_edges_from(nx.selfloop_edges(d))
    """
    for n in d.nodes: # self-edge
        if not d.has_edge(n, n):
            d.add_edge(n, n)
    """

    print(d)
    new_adj = nx.to_scipy_sparse_array(d, nodelist=sorted(d.nodes()))
    return new_adj

def ppr_overlay(nxg, k, add_local_edge=False, reversely=False, residual=False, exclude_overlay=False):

    d = nxg
    temp = []
    for _ in range(k):

        pr = nx.pagerank(d)
        #pr = nx.closeness_centrality(d)
        #pr = nx.global_reaching_centrality(d)
        #pr = nx.eigenvector_centrality(d)
        #pr = nx.laplacian_centrality(d)

        prlist = []
        for name, value in pr.items():
            prlist.append((name, value))
        prlist = sorted(prlist, key=lambda x: x[1])
        prlist.reverse()

        C = p_kpathcover(d, 1, directed=True, ordered_nodes=prlist)
        #C = p_kpathcover(d, 1, directed=True, ordered_nodes=None)
        d = overlay(d, C, directed=True)
        temp.append((d, C))

    d = nxg
    C = []
    for d_, C_ in temp:
        d = nx.compose(d, d_)
        C.extend(C_)
        C = list(set(C))
    C = sorted(C)

    """
    C = list(nxg.nodes())
    d = overlay(nxg, C, directed=True)
    d = nx.compose(d, nxg)
    """

    if add_local_edge:
        for v in C:
            visit = bounded_traverse(nxg, v, C, directed=True, reversely=reversely, ret_visit=True)
            for u in visit:
                if u != v:
                    if not d.has_edge(v, u):
                        d.add_edge(v, u)

    #d.remove_edges_from(nx.selfloop_edges(d))
    """
    for n in d.nodes: # self-edge
        if not d.has_edge(n, n):
            d.add_edge(n, n)
    """

    return d

def ppr_overlay(nxg, k, add_local_edge=False, reversely=False, residual=False, exclude_overlay=False):

    d = nxg
    temp = []
    for _ in range(k):

        #pr = nx.pagerank(d)
        pr = nx.closeness_centrality(d)
        #pr = nx.global_reaching_centrality(d)
        #pr = nx.eigenvector_centrality(d)
        #pr = nx.laplacian_centrality(d)

        prlist = []
        for name, value in pr.items():
            prlist.append((name, value))
        prlist = sorted(prlist, key=lambda x: x[1])
        #prlist.reverse()

        C = p_kpathcover(d, 1, directed=True, ordered_nodes=prlist)
        #C = p_kpathcover(d, 1, directed=True, ordered_nodes=None)
        d = overlay(d, C, directed=True)
        temp.append((d, C))

    d = nxg
    C = []
    for d_, C_ in temp:
        d = nx.compose(d, d_)
        C.extend(C_)
        C = list(set(C))
    C = sorted(C)

    """
    C = list(nxg.nodes())
    d = overlay(nxg, C, directed=True)
    d = nx.compose(d, nxg)
    """

    if add_local_edge:
        for v in C:
            visit = bounded_traverse(nxg, v, C, directed=True, reversely=reversely, ret_visit=True)
            for u in visit:
                if u != v:
                    if not d.has_edge(v, u):
                        d.add_edge(v, u)

    #d.remove_edges_from(nx.selfloop_edges(d))
    """
    for n in d.nodes: # self-edge
        if not d.has_edge(n, n):
            d.add_edge(n, n)
    """

    return d



if __name__ == "__main__":
    
    g = nx.DiGraph()
    g.add_edges_from([(1, 2), (1, 3), (1, 4), (3, 5), (2, 6), (7, 4), (8, 7), (9, 8), (10, 9), (11, 10)])
    #C = p_kpathcover(g, 3)
    #d = overlay(g, C)
    d = isset_overlay(g)
    print("*", d.nodes, d.edges)
