import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_3d_vector_field(vector_field):

    nx, ny, nz, _ = vector_field.shape
    x, y, z = np.meshgrid(np.arange(0,nx,1e-2), np.arange(0,ny,1e-2), np.arange(0,nz,1e-2), indexing='ij')

    u = vector_field[..., 0]
    v = vector_field[..., 1]
    w = vector_field[..., 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(x, y, z, u, v, w, length=0.5, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
def plot_vector_field(Ex, Ey, Ez, step=1):
    """
    Plots a 3D vector field using quiver plot.

    Parameters:
    - Ex, Ey, Ez: Arrays of shape (n, n, n) representing the x, y, and z components of the vector field.
    - step: Step size for sampling points in the grid to reduce plot density.
    """
    # Validate input shapes
    if Ex.shape != Ey.shape or Ex.shape != Ez.shape:
        raise ValueError("Ex, Ey, and Ez must have the same shape.")

    # Create a grid for the vector field
    n = Ex.shape[0]
    x, y, z = np.meshgrid(np.arange(0,n), np.arange(0,n), np.arange(0,n), indexing='ij')

    # Sample the field for plotting
    sampled_Ex = Ex[::step, ::step, ::step]
    sampled_Ey = Ey[::step, ::step, ::step]
    sampled_Ez = Ez[::step, ::step, ::step]

    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, sampled_Ex, sampled_Ey, sampled_Ez, length=0.5, normalize=True, color='b')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Vector Field')

    plt.show()
    
def load_vector_fields(path):
    loaded = np.load(path)
    return loaded['vector_field1'], loaded['vector_field2']

a,b = load_vector_fields('Fields/500.npz')
a,b = np.array(a),np.array(b)
# plot_3d_vector_field(a)
plot_vector_field(a[0],a[1],a[2])
# print(a.shape)