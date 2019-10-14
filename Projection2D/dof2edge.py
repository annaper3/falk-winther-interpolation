from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

# Computes two mappings: dof to edge and edge to dof

def dof2edge(mesh):

    V = FunctionSpace(mesh, 'N1curl', 1)
    N = V.dim()

    # Cell to edge map
    cell2edges = V.mesh().topology()(2, 1)
    # Cell to dof map
    cell2dofs = V.dofmap().cell_dofs

    dof2edge = np.zeros(mesh.num_edges(), dtype="int")
    edge2dof = np.zeros(mesh.num_edges(), dtype="int")
    # Iterate over cells, associating the edges to the dofs for that cell
    for c in range(mesh.num_cells()):
        # get the global edge numbers for this cell
        c_edges = cell2edges(c)
        # get the global dof numbers for this cell
        c_dofs = cell2dofs(c)
        # associate the edge numbers to the corresponding dof numbers and vice versa
        dof2edge[c_dofs] = c_edges
        edge2dof[c_edges] = c_dofs

    return dof2edge, edge2dof
