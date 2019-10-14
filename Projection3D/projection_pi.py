from fenics import *
import numpy as np
from compute_meshpatch import *
from compute_nodepatch import *
from parentmap_edge import *
from dof2edge import *

# This function returns the Falk-Winther projection pi^E_H(f) of a function f in H(curl).

def projection_pi(f, curl_f, mesh):
    mesh.init() #Initialize all entities
    coord = mesh.coordinates()
    d2e, e2d = dof2edge(mesh) #Compute dof to edge mapping
    P1_f = np.zeros(mesh.num_edges()) #First part of pi^E_H
    P2_f = np.zeros(mesh.num_edges()) #Second part of pi^E_H

    #Elements
    RT = FiniteElement('RT',tetrahedron,1) #Raviart-Thomas
    NE = FiniteElement('N1curl',tetrahedron,1) #Nedelec
    LA = FiniteElement('Lagrange',tetrahedron,1) #Lagrange
    DG = FiniteElement('DG',tetrahedron,0) #DG

    #Global spaces
    V_P1 = FunctionSpace(mesh, LA)
    V_NE = FunctionSpace(mesh, NE)

    for E in edges(mesh):

        #Compute meshpatch and nodal patch measures
        mesh_epatch, pm_node, pm_cell, npatch_meas = compute_meshpatch(mesh,E)
        #Find parent map for the edges
        pm_edge = parentmap_edge(mesh,mesh_epatch,pm_node)
        d2e_epatch, e2d_epatch = dof2edge(mesh_epatch) #Compute dof to edge mapping

        #Construct delta_z
        DG0 = FunctionSpace(mesh_epatch,DG) #DG0-space for indicator functions
        delta_z = Function(DG0)
        cell2dofs = DG0.dofmap().cell_dofs #Mapping from cell to dof
        z0 = np.zeros((mesh.num_cells(),2))
        #Create indicator function
        for i, node in enumerate(vertices(E)):
            v_cells = node.entities(3) #get neighbouring cells to node
            for c in v_cells:
                z0[c,i] = 1.0/npatch_meas[i]
        #Restrict to cells in edgepatch and construct z_2-z_1 (delta_z)
        z_patch = z0[pm_cell,1]-z0[pm_cell,0]
        #Reorder according to dofs and assign to delta_z
        z_patch_dofs = np.zeros(mesh_epatch.num_cells())
        for i in range(mesh_epatch.num_cells()):
            z_patch_dofs[i] = z_patch[cell2dofs(i)]
        delta_z.vector().set_local(z_patch_dofs)

        #Compute z_1
        RTNE = RT*NE #Mixed space
        W = FunctionSpace(mesh_epatch,RTNE)
        w1, w2 = TrialFunctions(W)
        tau1, tau2 = TestFunctions(W)
        z = Function(W)
        #Mixed formulation
        L = inner(div(w1),div(tau1))*dx + inner(tau1,curl(w2))*dx + inner(w1,curl(tau2))*dx
        F = inner(-delta_z,div(tau1))*dx
        #Dirichlet BC on RT and NE space
        bc_RT = DirichletBC(W.sub(0),Constant((0,0,0)), DomainBoundary())
        bc_NE = DirichletBC(W.sub(1),Constant((0,0,0)), DomainBoundary())
        #Solve system using numpy
        L = assemble(L)
        F = assemble(F)
        bc_RT.apply(L,F) #Puts zeros and ones in the matrix/rhs to enforce BC
        bc_NE.apply(L,F)
        L = L.array()
        F = F.get_local()
        x = np.linalg.lstsq(L,F,rcond=None)[0]
        #Assign x to z and obtain z_1 (first part of z)
        z.vector().set_local(x)
        z_1, z_2 = z.split(deepcopy=True)
        #Check that div(z_1) = -delta_z
        assert np.allclose(project(div(z_1), DG0).vector().get_local(),-delta_z.vector().get_local())

        #Apply M^1 to f
        M_1  = assemble(inner(f,z_1)*dx)
        P1_f[e2d[E.index()]] += M_1

        # Compute Q^1_{y,-} of f
        for i, node_index in enumerate(E.entities(0)):
            node = Vertex(mesh,node_index)
            #Compute Q^1_{y,-}
            mesh_npatch, pm_node_npatch, pm_cell_npatch = compute_nodepatch(mesh,node.index())
            V_P1_npatch = FunctionSpace(mesh_npatch, LA)
            v2d_npatch = vertex_to_dof_map(V_P1_npatch) #Mapping from vertex to dof
            Q_y = TrialFunction(V_P1_npatch)
            v = TestFunction(V_P1_npatch)
            L = assemble(inner(grad(Q_y),grad(v))*dx).array()
            F = assemble(inner(f,grad(v))*dx).get_local()
            C = assemble(inner(Q_y,1)*dx).get_local() #mean constraint
            #Set up saddle point system
            N = V_P1_npatch.dim()
            A = np.zeros((N+1,N+1))
            b = np.zeros(N+1)
            A[0:N, 0:N] = L
            A[N, 0:N] = C.transpose()
            A[0:N, N] = C
            b[0:N] = F
            #Solve system
            x = np.linalg.solve(A,b)
            assert np.allclose(A.dot(x),b)

            y = np.where(node.index() == pm_node_npatch)[0]
            P1_f[e2d[E.index()]] += (-1)**(i+1)*x[v2d_npatch[y]]


        #P1_f is done for E. Compute P2_f for E.
        S1_Qf_values = np.zeros(mesh.num_edges())

        #Compute Q^1_E(f)
        V_P1_epatch = FunctionSpace(mesh_epatch, LA)
        V_NE_epatch = FunctionSpace(mesh_epatch, NE)
        N_P1 = V_P1_epatch.dim()
        N_NE = V_NE_epatch.dim()
        Q = TrialFunction(V_NE_epatch)
        tau = TestFunction(V_P1_epatch)
        v = TestFunction(V_NE_epatch)
        S = assemble(inner(curl(Q),curl(v))*dx).array()
        T = assemble(inner(Q, grad(tau))*dx).array()
        F = assemble(inner(f,grad(tau))*dx).get_local()
        G = assemble(inner(curl_f,curl(v))*dx).get_local()
        K = np.zeros((N_P1 + N_NE, N_P1 + N_NE))
        rhs = np.zeros(N_P1 + N_NE)
        # Assemble saddle point system
        K[0:N_NE,0:N_NE] = S
        K[0:N_NE,N_NE:N_P1 + N_NE] = T.transpose()
        K[N_NE:N_P1 + N_NE,0:N_NE] = T
        rhs[0:N_NE] = G
        rhs[N_NE:N_NE + N_P1] = F
        x = np.linalg.solve(K,rhs)
        assert np.allclose(K.dot(x),rhs)
        Q_f = Function(V_NE_epatch)
        Q_f.vector().set_local(x[0:N_NE])

        #Apply M_1 to Q^1_E(f) for edge E
        M_1  = assemble(inner(Q_f,z_1)*dx)
        S1_Qf_values[e2d[E.index()]] += M_1

        # Compute Q^1_{y,-} of Q^1_E(f)
        for i, node_index in enumerate(E.entities(0)):
            node = Vertex(mesh,node_index)

            mesh_npatch, pm_node_npatch, pm_cell_npatch = compute_nodepatch(mesh,node.index())
            pm_edge_npatch = parentmap_edge(mesh,mesh_npatch,pm_node_npatch)
            V_P1_npatch = FunctionSpace(mesh_npatch, LA)
            v2d_npatch = vertex_to_dof_map(V_P1_npatch) #Mapping from vertex to dof
            V_NE_npatch = FunctionSpace(mesh_npatch,NE)
            d2e_npatch, e2d_npatch = dof2edge(mesh_npatch) #Compute dof to edge mapping

            #Restrict Q_f to nodepatch
            Q_f_omega_values = np.zeros(V_NE.dim())
            Q_f_omega_values[e2d[pm_edge[d2e_epatch]]] = Q_f.vector().get_local()
            Q_f_npatch = Function(V_NE_npatch)
            Q_f_npatch_values = np.zeros(V_NE_npatch.dim())
            Q_f_npatch_values = Q_f_omega_values[e2d[pm_edge_npatch[d2e_npatch]]]
            Q_f_npatch.vector().set_local(Q_f_npatch_values)

            #Compute Q^1_{y,-}
            Q_y = TrialFunction(V_P1_npatch)
            v = TestFunction(V_P1_npatch)
            L = assemble(inner(grad(Q_y),grad(v))*dx).array()
            F = assemble(inner(Q_f_npatch,grad(v))*dx).get_local()
            C = assemble(inner(Q_y,1)*dx).get_local() #mean constraint
            #Set up saddle point system
            N = V_P1_npatch.dim()
            A = np.zeros((N+1,N+1))
            b = np.zeros(N+1)
            A[0:N, 0:N] = L
            A[N, 0:N] = C.transpose()
            A[0:N, N] = C
            b[0:N] = F
            #Solve system
            x = np.linalg.solve(A,b)
            assert np.allclose(A.dot(x),b)

            y = np.where(node.index() == pm_node_npatch)[0]
            S1_Qf_values[e2d[E.index()]] += (-1)**(i+1)*x[v2d_npatch[y]]

        #Evaluate integral
        S1_Qf = Function(V_NE)
        S1_Qf.vector().set_local(S1_Qf_values)
        t = E.entities(0) #Edge nodes to define the tangent
        P2_f[e2d[E.index()]] = (Q_f(E.midpoint()) - S1_Qf(E.midpoint())).dot(coord[t[1]]-coord[t[0]])

    pi_f = Function(V_NE)
    pi_f.vector().set_local(P1_f+P2_f)
    return pi_f
