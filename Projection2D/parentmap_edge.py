from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

# Finds the parentmap for edges using the parentmap for nodes

def parentmap_edge(mesh,meshpatch,parentmap):

    # initialize meshes (if not already done)
    mesh.init()
    meshpatch.init()
    #to store result
    parentmap_edge = np.zeros(meshpatch.num_edges(), dtype="int")
    mesh_edges = np.zeros((mesh.num_edges(),2),dtype="int")

    # find the node of an edge
    for i, e in enumerate(edges(mesh)):
        mesh_edges[i,0] = e.entities(0)[0]
        mesh_edges[i,1] = e.entities(0)[1]

    # identify edges
    for i, e in enumerate(edges(meshpatch)):
        vert = e.entities(0)
        ind = np.where((mesh_edges[:,0] == parentmap[vert[0]]) & (mesh_edges[:,1] == parentmap[vert[1]]))[0]
        if ind.size==0:
            ind = np.where((mesh_edges[:,0] == parentmap[vert[1]]) & (mesh_edges[:,1] == parentmap[vert[0]]))[0]
        parentmap_edge[i] = ind[0]
    return parentmap_edge
