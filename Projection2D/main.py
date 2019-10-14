from fenics import *
import numpy as np
from projection_pi import *

# This code computes uses the projection_pi function to compute the Falk-Winther
# projection of a function f

mesh = UnitSquareMesh(2**2,2**2)
f = Expression(('0.0','x[0]*(x[0]-1)'), degree = 2)
curl_f = Expression('2*x[0]-1', degree = 1)

#Compute projection
pi_f = projection_pi(f, curl_f, mesh)

#print error in H(curl)-norm
print sqrt(assemble(inner(f-pi_f,f-pi_f)*dx + inner(curl_f-curl(pi_f),curl_f-curl(pi_f))*dx))
