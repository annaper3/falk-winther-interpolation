from fenics import *
import numpy as np

# Compute the edgepatch given an edge index

def compute_meshpatch(mesh, e):
    #find nodepatch and edgepatch nodes/cells
    npatch_meas = [0, 0] #Measure of nodepatch
    epatch_nodes, epatch_cells = [], [] #incl. boundary nodes
    npatch_nodes, npatch_cells = [], []
    for i, vert in enumerate(vertices(e)):
        nbhd_nodes = [Edge(mesh, j).entities(0) for j in vert.entities(1)]
        #nbhd_edges = vert.entities(1)
        nbhd_cells = vert.entities(2)
        nbhd_nodes = np.array(nbhd_nodes).flatten()
        nbhd_cells = np.array(nbhd_cells).flatten()
        npatch_nodes.append(np.unique(nbhd_nodes))
        npatch_cells.append(np.unique(nbhd_cells))
        for c in nbhd_cells:
            npatch_meas[i] += Cell(mesh, c).volume()
    epatch_nodes = np.unique(np.concatenate(npatch_nodes, axis = 0))
    epatch_cells = np.unique(np.concatenate(npatch_cells, axis = 0))

    tdim = mesh.topology().dim()
    gdim = mesh.geometry().dim()

    #Create coarse edge patch with mesh editor (preserves orientation)
    editor = MeshEditor()
    mesh_epatch = Mesh()
    editor.open(mesh_epatch, 'triangle', tdim, gdim)
    editor.init_vertices(epatch_nodes.size)  # number of vertices
    editor.init_cells(epatch_cells.size)     # number of cells
    used_vertices = []
    local_index_v = np.zeros(3, dtype = int)
    global_to_local = {}
    epatch_cells_ref = []
    epatch_nodes_ref = []

    for i,node in enumerate(epatch_nodes):
        global_to_local[node] = i

    for i, cell in enumerate(epatch_cells):
        c = Cell(mesh,cell)
        #Add coarse cells and nodes to mesh
        for j, node in enumerate(c.entities(0)):
            v = Vertex(mesh, node)
            if node not in used_vertices:
                editor.add_vertex(global_to_local[node], v.point())
                used_vertices.append(node)
            local_index_v[j] = global_to_local[node]
        editor.add_cell(i,local_index_v[0],local_index_v[1],local_index_v[2])

    editor.close()
    mesh_epatch.init()

    return mesh_epatch, epatch_nodes, epatch_cells, npatch_meas
