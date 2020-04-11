# -*- coding: utf-8 -*-
import numpy as np

# The polyhedron greedy algorithm [Edmonds '71]
# Implementation by Andreas Krause
# Port to Python: Gavin Smith (gavin.smith@nottingham.ac.uk)
#
# function x = sfo_polyhedrongreedy(F,V,w)
# F: Submodular function
# V: index set
# w: weight vector, w(i) is weight of V(i)
#
# Example:
#   x = sfo_polyhedrongreedy(@sfo_fn_example,1:2,sfo_charvector(1:2,1))

import numpy as np
from numpy import linalg
def sfo_polyhedrongreedy(F,V,w_in):
    if len(w_in.shape) == 2 and w_in.shape[1] != 1:
        raise Exception('Python translation error')
    n = V.shape[1]
    w = np.flip(np.sort(w_in,kind='stable',axis=0))
    I = np.flip(np.argsort(w_in, kind='stable',axis=0))
    x = np.zeros((1,n))
    A = []
    Fold = F(np.asarray([]))
    for i in range(n):
        # Anew = [A V(I(i))];
        Anew = np.hstack((A,V[0,I[i]]))
        x[0,I[i]] = F(Anew)-Fold
        A = Anew
        Fold = Fold + x[0,I[i]]

    return x


# returns characteristic vector of A
# Author: Andreas Krause (krausea@gmail.com)
# Port to Python: Gavin Smith (gavin.smith@nottingham.ac.uk)
#
# function w = sfo_charvector(V,A)
# V: index set
# A: subset of V
#
# Example: sfo_charvector([2,1,4],[4,2])=[1 0 1]

def sfo_charvector(V,A):
    return np.in1d(V, A)

# Finding the minimum of a submodular function using Wolfe's min norm point
# algorithm [Fujishige '91]
# Implementation by Andreas Krause (krausea@gmail.com)
# Port to Python: Gavin Smith (gavin.smith@nottingham.ac.uk)
#
# function A = sfo_min_norm_point(F,V, opt)
# F: Submodular function
# V: index set
# opt (optional): option struct of parameters, referencing:
#
# minnorm_init: starting guess for optimal solution
# minnorm_stopping_thresh: stopping threshold for search
# minnorm_tolerance: numerical tolerance
# minnorm_callback: callback routine for visualizing intermediate solutions
#
# Returns: optimal solution A, bound on suboptimality
#
# Example: A = sfo_min_norm_point(sfo_fn_example,1:2);

#function [A,subopt] = sfo_min_norm_point(F,V, opt) 
def sfo_min_norm_point(F,V, Ainit = np.asarray([]), eps =1e-10, TOL = 1e-10, verbosity_level = 1 ):

    n=len(V)
    V = np.zeros((1,n)) + V
    # step 1: initialize by picking a point in the polytope
    wA = sfo_charvector(V,Ainit)
    xw = sfo_polyhedrongreedy(F,V,wA)
    S = xw.conj().transpose()
    xhat = xw.conj().transpose()
    Abest = -1
    Fbest = np.inf
    while True:
        # step 2: 
        if np.linalg.norm(xhat)<TOL: #snap to zero
            xhat = np.zeros((1,xhat.shape))
        

        # get phat by going from xhat towards the origin until we hit
        # boundary of polytope P
        phat = sfo_polyhedrongreedy(F,V,-xhat).conj().transpose()
        S = np.hstack((S, phat))
        
        # check current function value
        Fcur = F(V[0,xhat.flatten()<0])
        if Fcur<Fbest:
            Fbest = Fcur
            Abest = V[0,xhat.flatten()<0] #this gives the unique minimal minimizer
      
        # get suboptimality bound
        subopt = Fbest-np.sum(xhat[xhat<0])
        if verbosity_level>0:
            print('suboptimality bound: {} <= min_A F(A) <= F(A_best) = {}; delta<={}'.format(Fbest-subopt,Fbest,subopt))
        
        
        # if abs(xhat'*phat - xhat'*xhat)<TOL || (subopt<eps)
        # @ is matrix multiplication
        gcheck = xhat.conj().transpose() @ phat - xhat.conj().transpose() @ xhat


        if (abs(gcheck)<TOL) or (subopt<eps):
            # we are done: xhat is already closest norm point
            if abs(xhat.conj().transpose() @ phat - xhat.conj().transpose() @ xhat)<TOL:
                subopt = 0
            
            A = Abest
            break
        
        
        # here's some code just for outputting the current state
        # can be used to visualize progress in the Ising model (tutorial)
        # if isfield(opt,'minnorm_callback') #do something with current state
        #     if isa(opt.minnorm_callback,'function_handle')
        #         opt.minnorm_callback(Abest);
        #     end
        # end
        
        xhat,S = sfo_min_norm_point_update_xhat(xhat, S, TOL)
    
    if verbosity_level>0:
        print('suboptimality bound: {} <= min_A F(A) <= F(A_best) = {}; delta<={}'.format(Fbest-subopt,Fbest,subopt))
    

    return A, subopt

## Helper function for updating xhat

#function [xhat,S] = sfo_min_norm_point_update_xhat(xhat, S, TOL)

def sfo_min_norm_point_update_xhat(xhat, S, TOL):

    while True:
        # step 3: Find minimum norm point in affine hull spanned by S

        # S0 = S(:,2:end)-S(:,ones(1,size(S,2)-1)); % subspace after translating by S(:,1)
        S0 = S[:,1:] - np.tile(S[:,0].reshape(S.shape[0],1),S[:,1:].shape[1])
        
        # may need linalg.lstsq instead
        # y = S(:,1)- S0*(
        #                   ( S0'*S0 )\(  S0'*S(:,1)  )
        #                 ); %now y is min norm
        y = S[:,0].reshape(S.shape[0],1) - S0 @ np.asarray([( linalg.solve(S0.conj().transpose()@S0, S0.conj().transpose()@S[:,0]) ) ]).T
            
            
            #S[:,0]- S0@((),(S0.conj().transpose()@S[:,0]))) #now y is min norm

        #get representation of y in terms of S. Enforce
        #affine combination (i.e., sum(mu)==1)
        # mu = [S;ones(1,size(S,2))]\[y;1];
        mu = linalg.lstsq( np.vstack((S, np.ones((1,S.shape[1])) )) , np.concatenate( (y,np.ones((1,y.shape[1]))) ),rcond=None,  )[0]
            
        # y is written as positive convex combination of S <==> y in
        # conv(S)
        if np.sum(mu < -TOL)==0 and abs( np.sum(mu)-1 ) < TOL:
        # y is in the relative interior of S
            xhat = y
            break
        

        # step 4: #project y back into polytope

        #get representation of xhat in terms of S; enforce that we get
        #affine combination (i.e., sum(lambda)==1)
        #lambda = [S;ones(1,size(S,2))]\[xhat;1];
        #lambda_var = np.linalg.solve( np.concatenate((S,np.ones(1,S.shape[1]))), np.concatenate((xhat,1)) )
        lambda_var = linalg.lstsq( np.vstack((S, np.ones((1,S.shape[1])) )) , np.concatenate( (xhat,np.ones((1,xhat.shape[1]))) ) ,rcond=None, )[0]
           

        # now find z in conv(S) that is closest to y
        bounds = lambda_var/(lambda_var-mu)
        bounds = bounds[bounds>TOL]
        beta = np.min(bounds)
        z = (1-beta) * xhat + beta * y

        gamma = (1-beta) * lambda_var + beta * mu; # find relevant coordinates
        S = S[:,(gamma)>TOL]
        xhat = z    

    return xhat,S
