# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:47:33 2021

@author: Mateo
"""

import matplotlib.pyplot as plt
import numpy as np

# %% Define a plotting function
def plotFuncs3D(functions,bottom,top,N=300,titles = None, ax_labs = None):
    '''
    Plots 3D function(s) over a given range.

    Parameters
    ----------
    functions : [function] or function
        A single function, or a list of functions, to be plotted.
    bottom : (float,float)
        The lower limit of the domain to be plotted.
    top : (float,float)
        The upper limit of the domain to be plotted.
    N : int
        Number of points in the domain to evaluate.
    titles: None, or list of string
        If not None, the titles of the subplots

    Returns
    -------
    none
    '''
    
    if type(functions)==list:
        function_list = functions
    else:
        function_list = [functions]
    
    nfunc = len(function_list)
    
    # Initialize figure and axes
    fig = plt.figure(figsize=plt.figaspect(1.0/nfunc))
    # Create a mesh
    x = np.linspace(bottom[0],top[0],N,endpoint=True)
    y = np.linspace(bottom[1],top[1],N,endpoint=True)
    X,Y = np.meshgrid(x, y)
    
    for k in range(nfunc):
        
        # Add axisplt
        ax = fig.add_subplot(1, nfunc, k+1, projection='3d')
        #ax = fig.add_subplot(1, nfunc, k+1)
        Z = function_list[k](X,Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        if ax_labs is not None:
            ax.set_xlabel(ax_labs[0])
            ax.set_ylabel(ax_labs[1])
            ax.set_zlabel(ax_labs[2])
        #ax.imshow(Z, extent=[bottom[0],top[0],bottom[1],top[1]], origin='lower')
        #ax.colorbar();
        if titles is not None:
            ax.set_title(titles[k]);
            
        ax.set_xlim([bottom[0], top[0]])
        ax.set_ylim([bottom[1], top[1]])
        
    plt.show()

def plotSlices3D(functions,bot_x,top_x,y_slices,N=300,y_name = None,
                 titles = None, ax_labs = None):

    if type(functions)==list:
        function_list = functions
    else:
        function_list = [functions]
    
    nfunc = len(function_list)
    
    # Initialize figure and axes
    fig = plt.figure(figsize=plt.figaspect(1.0/nfunc))
    
    # Create x grid
    x = np.linspace(bot_x,top_x,N,endpoint=True)
    
    for k in range(nfunc):
        ax = fig.add_subplot(1, nfunc, k+1)
                
        for y in y_slices:
            
            if y_name is None:
                lab = ''
            else:
                lab = y_name + '=' + str(y)
            
            z = function_list[k](x, np.ones_like(x)*y)
            ax.plot(x,z, label = lab)
            
        if ax_labs is not None:
            ax.set_xlabel(ax_labs[0])
            ax.set_ylabel(ax_labs[1])
            
        #ax.imshow(Z, extent=[bottom[0],top[0],bottom[1],top[1]], origin='lower')
        #ax.colorbar();
        if titles is not None:
            ax.set_title(titles[k]);
            
        ax.set_xlim([bot_x, top_x])
        
        if y_name is not None:
            ax.legend()
        
    plt.show()

def plotSlices4D(functions,bot_x,top_x,y_slices,w_slices,N=300,
                 slice_names = None, titles = None, ax_labs = None):

    if type(functions)==list:
        function_list = functions
    else:
        function_list = [functions]
    
    nfunc = len(function_list)
    nws   = len(w_slices)
    
    # Initialize figure and axes
    fig = plt.figure(figsize=plt.figaspect(1.0/nfunc))
    
    # Create x grid
    x = np.linspace(bot_x,top_x,N,endpoint=True)
    
    for j in range(nws):
        w = w_slices[j]
        
        for k in range(nfunc):
            ax = fig.add_subplot(nws, nfunc, j*nfunc + k+1)
                    
            for y in y_slices:
                
                if slice_names is None:
                    lab = ''
                else:
                    lab = slice_names[0] + '=' + str(y) + ',' + \
                          slice_names[1] + '=' + str(w)
                
                z = function_list[k](x, np.ones_like(x)*y, np.ones_like(x)*w)
                ax.plot(x,z, label = lab)
                
            if ax_labs is not None:
                ax.set_xlabel(ax_labs[0])
                ax.set_ylabel(ax_labs[1])
                
            #ax.imshow(Z, extent=[bottom[0],top[0],bottom[1],top[1]], origin='lower')
            #ax.colorbar();
            if titles is not None:
                ax.set_title(titles[k]);
                
            ax.set_xlim([bot_x, top_x])
            
            if slice_names is not None:
                ax.legend()
        
    plt.show()