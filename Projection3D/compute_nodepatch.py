from fenics import *
import numpy as np

# Compute the nodepatch given a node index

def compute_nodepatch(mesh, vertex_index):
    #find nodepatch nodes/cells
    npatch_nodes, npatch_cells = [], []
    vert = Vertex(mesh,vertex_index)
    nbhd_nodes = [Edge(mesh, j).entities(0) for j in vert.entities(1)]
    nbhd_cells = vert.entities(3)
    nbhd_nodes = np.array(nbhd_nodes).flatten()
    nbhd_cells = np.array(nbhd_cells).flatten()
    npatch_nodes.append(np.unique(nbhd_nodes))
    npatch_cells.append(np.unique(nbhd_cells))
    npatch_nodes = np.array(npatch_nodes).flatten()
    npatch_cells = np.array(npatch_cells).flatten()


    tdim = mesh.topology().dim()
    gdim = mesh.geometry().dim()

    #Create coarse edge patch with mesh editor (preserves orientation)
    editor = MeshEditor()
    mesh_npatch = Mesh()
    editor.open(mesh_npatch, 'tetrahedron', tdim, gdim)
    editor.init_vertices(npatch_nodes.size)  # number of vertices
    editor.init_cells(npatch_cells.size)     # number of cells
    used_vertices = []
    local_index_counter = 0
    global_to_local = {}
    global_to_local_ref = {}
    local_index_v = np.zeros(4, dtype = int)
    npatch_cells_ref = []
    npatch_nodes_ref = []

    for i,node in enumerate(npatch_nodes):
        global_to_local[node] = i

    for i, cell in enumerate(npatch_cells):
        c = Cell(mesh,cell)

        #Add coarse cells and nodes to mesh
        for j, node in enumerate(c.entities(0)):
            v = Vertex(mesh, node)
            if node not in used_vertices:
                editor.add_vertex(global_to_local[node], v.point())
                used_vertices.append(node)
            local_index_v[j] = global_to_local[node]
        editor.add_cell(i,local_index_v[0],local_index_v[1],local_index_v[2],local_index_v[3])
    editor.close()
    mesh_npatch.init()

    return mesh_npatch, npatch_nodes, npatch_cells
