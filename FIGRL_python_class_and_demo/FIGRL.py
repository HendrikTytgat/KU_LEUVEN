# -*- coding: utf-8 -*-
"""
Created on 10/08/2020 08:45:24 2020

@author: Hendrik

"""


import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.sparse
from scipy.sparse import coo_matrix
from scipy.sparse import csr
import collections

class FIGRL:
    
    """
    This class initializes a Fast Inductive Graph Representation Learning framework
    
    Parameters
    ----------
    embedding_size : int
        The desired size of the resulting embeddings
    intermediate_dimension : int
        The dimension of the matrix sketch M 
    """
    
   
    def __init__(self, embedding_size, intermediate_dimension):

        self.embedding_size = embedding_size
        self.intermediate_dimension = intermediate_dimension
        self.St = None
        self.V = None
        self.sigma = None
        
    def train(self, train_graph, S=None):

        """
        
        This function trains a figrl model.
        It returns the trained figrl model and a pandas datarame containing the embeddings generated for the train nodes.
        
        Parameters
        ----------
        train_graph : NetworkX Object
            The graph on which the training step is done on.
        S : numpy randn matrix of size #of nodes in train_graph
        """
        A = nx.adjacency_matrix(train_graph)
        n,m = A.shape
        diags = A.sum(axis=1).flatten()
        D = scipy.sparse.spdiags(diags, [0], n, n, format='csr')

        with scipy.errstate(divide='ignore'):
           diags_sqrt = 1.0/np.lib.scimath.sqrt(diags)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        DH = scipy.sparse.spdiags(diags_sqrt, [0], n, n, format='csr')

        Normalized_random_walk = DH.dot(A.dot(DH))
        if S is None:
            S = np.random.randn(n, self.intermediate_dimension) / np.sqrt(self.intermediate_dimension)
            np.savetxt("S_train_matrix.csv", S, delimiter=",")
        #S = np.array(pd.read_csv('S_train_matrix.csv', header=None))

        C = Normalized_random_walk.dot(S)

        from scipy import sparse
        sC = sparse.csr_matrix(C)

        U, self.sigma, self.V = scipy.sparse.linalg.svds(sC, k=self.embedding_size, tol=0,which='LM')
        self.V = self.V.transpose()
        self.sigma = np.diag(self.sigma)
        
        figrl_train_emb = pd.DataFrame(U)
        figrl_train_emb = figrl_train_emb.set_index(figrl_train_emb.index)
        
        self.sigma = np.array(self.sigma)
        self.V = np.array(self.V)
        self.St = np.array(S)
        return figrl_train_emb
    
    def create_inductive_dict(self, inductive_data ,list_connected_node_types):
        
        """
        This creates a collection of inductive node as key and values their interaction with other nodes in inductive step.
        
        Parameters
        ----------
        inductive_data : pandas Dataframe
            The row defines the incoming node, in the columns the different node types that can be connected to.

        """
        
        inductive_dict = {}
        for node in inductive_data.index:
            for connected_node in list_connected_node_types:
                if inductive_data.loc[node].connected_node != None:
                    inductive_dict[node].append(inductive_data.loc[node].connected_node)  
        inductive_dict = collections.OrderedDict(sorted(inductive_dict.items()))
        return inductive_dict
        
    def inductive_step(self, graph, inductive_dict, maxid, inductive_index):
 
        """
        This function generates embeddings for unseen nodes using a trained figrl model.
        It returns the embeddings for these unseen nodes. 
        
        Parameters
        ----------
        graph : NetworkX Object
            The graph on which FIGRL is deployed.
        U : The training embeddings
        
        maxid: int
            The maximum integer ID for the training and inductive set
            
        inductive_index: RangeIndex
            The inductive indexes for the embeddings
        """
    
        def get_vector(inductive_dict, train_degrees, max_id):
            print("creating sparse vector matrix")
            row  = []
            col  = []
            data = []
            i = 0
            for node, v in inductive_dict.items():
                for n in v:
                    if n is not None:
                        row.append(i)
                        col.append(n)
                    if n > max_id:
                        max_id = int(n)
                    #calculate value
                    inductive_degree = len([x for x in v if x != None])
                    value = 1/np.sqrt(inductive_degree)
                    value = value * (1/np.sqrt(train_degrees[n]))
                    data.append(value)
                i+=1
            row = np.array(row)
            col = np.array(col)
            data = np.array(data)
            return coo_matrix((data, (row, col)), shape=(len(inductive_dict), max_id+1))
        
        degrees = nx.degree(graph)
        train_degrees = dict(degrees)
        train_degrees = collections.OrderedDict(sorted(train_degrees.items()))
        
        v = get_vector(inductive_dict, train_degrees, maxid)
        
        S = np.random.randn(maxid+1, self.intermediate_dimension) / np.sqrt(self.intermediate_dimension)
        
        inductive_degrees = []

        for l in inductive_dict.values():
            x = 0
            for i in l:
                if i is not None:
                    x+=1
            inductive_degrees.append(x)
    

        sqrt_d_inv = np.array([1/np.sqrt(degree)  if degree > 0 else 0 for degree in inductive_degrees])
        sqrt_d_inv = scipy.sparse.spdiags(sqrt_d_inv,0, sqrt_d_inv.size, sqrt_d_inv.size)

        p = v.dot(S)
        U =(p.dot(self.V)).dot(np.linalg.inv(self.sigma))
        U = sqrt_d_inv.dot(U)
        
        figrl_inductive_emb = pd.DataFrame(U, index = inductive_index)
    
        return figrl_inductive_emb    
    
