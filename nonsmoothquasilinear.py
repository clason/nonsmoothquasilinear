"""
Semismooth Newton method for solving the optimal control problem
    ½‖y-yᵈ‖² + α½‖u‖²  s.t.  -div((1+|y|)∇y) = u, y = 0 on ∂Ω.

For details, see
Christian Clason, Vu Huu Nhu, Arnd Rösch:
Optimal control of a non-smooth quasilinear elliptic equation,
arXiv:1810.08007
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from dolfin import *

# problem parameters

parameters['linear_algebra_backend'] = 'Eigen'
def assemble_csr(a,bc=None):
    """assemble bilinear form a to SciPy matrix"""
    A = assemble(a)
    if bc:
        bc.apply(A)
    row,col,val = as_backend_type(A).data()
    return sp.csr_matrix((val,col,row))


def interpolate_V(z,V):               
    """create DOLFIN function from Numpy array"""
    z1 = Function(V)
    z1.vector().set_local(z)
    return z1


def mesh_space_functions(N):
    """setup FE grid, function spaces"""
    mesh = UnitSquareMesh(N,N)
    V    = FunctionSpace(mesh,'CG',1)
    X    = interpolate(Expression("x[0]",degree=1),V).vector().get_local()
    Y    = interpolate(Expression("x[1]",degree=1),V).vector().get_local()
    u    = TrialFunction(V)
    v    = TestFunction(V)
    n    = V.dim()                         # number of nodes
    return mesh,V,X,Y,u,v,n


def plot_fun(Z,title,X,Y):
    """plot function"""
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.cla()
    surf = ax.plot_trisurf(X, Y, Z, 
           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(1.*min((Z)), 1.*max((Z)))
    ax.zaxis.set_major_locator(LinearLocator(4))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%1.0e'))
    ax.set_title(title)
    plt.draw()
    plt.pause(.001)


def F(phi,psi,yd,alpha,v,V,A,M,bc):
    """evaluate relaxed optimality system at iterate"""
    z1   = A*psi + 1./alpha*M*phi
    psi0 = pow(1.+2*np.absolute(psi),0.5)
    f1   = (-1. + psi0)/(psi0)*np.sign(psi)
    f2   = 1./psi0
    F2   = -interpolate_V(f1 - np.multiply(yd,f2),V)*v*dx
    z2   = assemble(F2); bc.apply(z2)
    z2   = z2[:] + (A*phi)
    return np.concatenate([z1,z2])


def Newton_Jac(phi,psi,yd,alpha,u,v,V,A,M,bc):
    """evaluate Newton derivative at iterate"""
    psi1    = pow(1.+2* np.absolute(psi),1.5)
    hk      = ((1.-yd)*(psi >=0.).astype(float) + \
               (1.+ yd)*(psi < 0.).astype(float))/psi1
    dF2_psi = -interpolate_V(hk,V)*u*v*dx
    J22     = assemble_csr(dF2_psi,bc)
    DNF     = sp.bmat([[(1./alpha)*M, A], [A, J22]])
    return sp.csr_matrix(DNF)


def SSN_method(yd,alpha,u,v,V,n,A,M,bc):
    """solve relaxed optimality system using semismooth Newton method"""
    phik, psik = np.zeros(n), np.zeros(n)
    AC_set     = (psik >= 0).astype(float)
    maxit      = 25    # maximal number of Newton steps
    SSNit      = 0
    converged  = False
    print("solving the optimality system")
    while not converged:
        if SSNit >= maxit:
            break
        SSNit += 1
        rhs   = -F(phik,psik,yd,alpha,v,V,A,M,bc)
        nrhs  = np.linalg.norm(rhs)
        print("%s \t %1.3e" % ("residual", nrhs))
        dX = spsolve(Newton_Jac(phik,psik,yd,alpha,u,v,V,A,M,bc),rhs)
        dphi, dpsi = np.split(dX,2)
        phik += dphi
        psik += dpsi
        AC_set_new = (psik >= 0).astype(float)
        converged  = np.array_equal(AC_set,AC_set_new)
        AC_set     = AC_set_new
    return phik, psik, SSNit


def factor(nzero,X,Y): 
    """factors for constructing exact solution ybar"""
    factor2 = ((X<=nzero).astype(float))     
    factor0 = pow(X,4)*(pow(X-nzero,4) +2*pow(X-nzero,5))  
    factor1 = pow(np.sin(np.pi*Y),3)                      
    return factor0.ravel(),factor1.ravel(),factor2.ravel()


def der_factor(nzero,X,Y):           
    """first derivatives of factors"""
    der1A  = 4*pow(X,3)*(pow(X-nzero,4) +2*pow(X-nzero,5))
    der1A += pow(X,4)*(4*pow(X-nzero,3) +2*5*pow(X-nzero,4))
    der1B  = 3*np.pi*pow(np.sin(np.pi*Y),2)*np.cos(np.pi*Y)
    return der1A.ravel(), der1B.ravel()


def der2_factor(nzero,X,Y):         
    """second derivatives of factors"""
    der2A  = 12*pow(X,2)*(pow(X-nzero,4) +2*pow(X-nzero,5))
    der2A += 2*4*pow(X,3)*(4*pow(X-nzero,3) +2*5*pow(X-nzero,4))
    der2A += pow(X,4)*(12*pow(X-nzero,2) +2*5*4*pow(X-nzero,3))
    der2B  = 3*pow(np.pi,2)*np.sin(np.pi*Y)*(2-3*pow(np.sin(np.pi*Y),2))
    return der2A.ravel(), der2B.ravel()


def der3_factor(nzero,X,Y):         
    """third derivatives of factors"""
    der3A  = 12*2*X*(pow(X-nzero,4) +2*pow(X-nzero,5))
    der3A += 3*12*pow(X,2)*(4*pow(X-nzero,3) +2*5*pow(X-nzero,4))
    der3A += 3*4*pow(X,3)*(4*3*pow(X-nzero,2) +2*5*4*pow(X-nzero,3))
    der3A += pow(X,4)*(12*2*(X-nzero) +2*5*4*3*pow(X-nzero,2))
    der3B  = 3*pow(np.pi,3)*np.cos(np.pi*Y)*(2-9*pow(np.sin(np.pi*Y),2))
    return der3A.ravel(),der3B.ravel()


def der4_factor(nzero,X,Y):       
    """fourth derivatives of factors"""
    der4A  = 12*2*(pow(X-nzero,4) +2*pow(X-nzero,5)) + \
            12*2*X*(4*pow(X-nzero,3) +2*5*pow(X-nzero,4))
    der4A += 3*12*2*X*(4*pow(X-nzero,3) +2*5*pow(X-nzero,4)) + \
            3*12*pow(X,2)*(4*3*pow(X-nzero,2) +2*5*4*pow(X-nzero,3))
    der4A += 3*4*3*pow(X,2)*(4*3*pow(X-nzero,2) +2*5*4*pow(X-nzero,3)) + \
            3*4*pow(X,3)*(4*3*2*(X-nzero) +2*5*4*3*pow(X-nzero,2))
    der4A += 4*pow(X,3)*(12*2*(X-nzero) +2*5*4*3*pow(X-nzero,2)) + \
            pow(X,4)*(12*2 +2*5*4*3*2*(X-nzero))
    der4B  = 3*pow(np.pi,4)*np.sin(np.pi*Y)*(7-27*pow(np.cos(np.pi*Y),2))
    return der4A.ravel(),der4B.ravel()


def construct_example(nzero,alpha,X,Y):
    """construct exact solution, data for example"""
    factors = factor(nzero,X,Y)
    # exact state
    ybar = factors[0]*factors[1]*factors[2]

    # exact control
    derfactors  = der_factor(nzero,X,Y)
    der2factors = der2_factor(nzero,X,Y)

    d1yx1 = derfactors[0]*factors[1]*factors[2]
    d1yx2 = factors[0]*derfactors[1]*factors[2]
    d2yx1 = der2factors[0]*factors[1]*factors[2]
    d2yx2 = der2factors[1]*factors[0]*factors[2]
    Del2  = d2yx1 + d2yx2
    ubar  = -(np.sign(ybar))*(d1yx1*d1yx1+d1yx2*d1yx2) \
            -(1.+np.absolute(ybar))*Del2

    # exact adjoint
    phibar = -alpha*ubar

    # data
    der3factors = der3_factor(nzero,X,Y)
    der4factors = der4_factor(nzero,X,Y)
    d2yx1x2     = derfactors[0]*derfactors[1]*factors[2]
    d3yx1       = der3factors[0]*factors[1]*factors[2]
    d3yx2       = der3factors[1]*factors[0]*factors[2]
    d3yx1_2x2   = der2factors[0]*derfactors[1]*factors[2]
    d3yx1x2_2   = der2factors[1]*derfactors[0]*factors[2]
    d4yx1       = der4factors[0]*factors[1]*factors[2]
    d4yx2       = der4factors[1]*factors[0]*factors[2]
    d4yx1_2x2_2 = der2factors[0]*der2factors[1]*factors[2]
    Del_u       = 3*(pow(d2yx1,2)+pow(d2yx2,2)) + 4*(d1yx1*d3yx1+d1yx2*d3yx2) \
                  + 4*pow(d2yx1x2,2) + 4*(d1yx1*d3yx1x2_2+d1yx2*d3yx1_2x2) \
                  + 2*d2yx1*d2yx2
    Del_u      *= -(np.sign(ybar))
    Del_u      -= (1. + np.absolute(ybar))*(d4yx1 +2*d4yx1_2x2_2 + d4yx2)
    y_d         = ybar - alpha*(1. + np.absolute(ybar))*Del_u
     
    # exact coefficient
    psibar = ybar*(1. + 0.5*np.absolute(ybar))

    return ybar, ubar, phibar, psibar, y_d


def run_example(N,alpha,beta):
    """set up and solve optimal control problem"""
    mesh,V,X,Y,u,v,n = mesh_space_functions(N)

    # assemble finite element stiffness, mass matrix
    bc = DirichletBC(V, Constant(0.0), lambda x,on_boundary: on_boundary)
    M  = assemble_csr(u*v*dx,bc)        # mass matrix
    A  = assemble_csr(dot(grad(u),grad(v))*dx,bc)
    
    # construct exact solution, data
    ybar,ubar, phibar,psibar,yd = construct_example(beta,alpha,X,Y)
    max_yd = np.amax(np.absolute(yd))

    ybar_func    = interpolate_V(ybar,V)
    phibar_func  = interpolate_V(phibar,V)
    psibar_func  = interpolate_V(psibar,V)

    phi,psi,SSNit = SSN_method(yd,alpha,u,v,V,n,A,M,bc)
    y = (-1. + pow(1.+2*np.absolute(psi),0.5))*np.sign(psi)

    y_func    = interpolate_V(y,V)
    phi_func  = interpolate_V(phi,V)
    rel_err_y   = errornorm(y_func,ybar_func,'H10')/norm(ybar_func,'H10')
    rel_err_phi = errornorm(phi_func,phibar_func,'H10')/norm(phibar_func,'H10')

    print("%-50s  %d" %("N", N))
    print("%-50s  %e " %("alpha", alpha))
    print("%-50s  %0.2f" %("beta", beta))
    print("%-50s  %d" %("#SSN it", SSNit))
    print("%-50s  %1.3e" %("parameter h",mesh.hmax()))
    print("%-50s  %1.3e" %("max y_d", max_yd))
    print("%-50s  %1.3e" %("relative H10 error of state", rel_err_y))
    print("%-50s  %1.3e" %("relative H10 error of adjoint", rel_err_phi))

    plot_fun(y,"y_h",X,Y)
    plot_fun(phi,"phi_h",X,Y)
    plt.show()


if __name__ == '__main__':
    """ Test routine if called as script """
    N     = 100
    beta  = 0.8
    alpha = 1.e-6
    run_example(N,alpha,beta)
