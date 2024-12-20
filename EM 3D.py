import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numba
from copy import deepcopy
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random

# Constants
epsilon_0 = 8.854187817e-12  # Vacuum permittivity in SI units (F/m)
mu_0 = 1.25663706127e-6  # Vacuum permeability in SI units (N/A**2)
elementary_charge = 1.602e-19
electron_mass = 9.109e-31

# testing
# epsilon_0 = 1
# mu_0 = 1
# elementary_charge = 1
# electron_mass = 1e-5

@numba.jit(nopython=True)
def gradient(arr, spacing, axis):
    # Create an empty array to store the np.gradient
    grad = np.zeros_like(arr)

    if axis == 0:
        # Compute np.gradient along the first axis (axis=0)
        grad[1:-1, :, :] = (arr[2:, :, :] - arr[:-2, :, :]) / (2 * spacing)
        grad[0, :, :] = (arr[1, :, :] - arr[0, :, :]) / spacing
        grad[-1, :, :] = (arr[-1, :, :] - arr[-2, :, :]) / spacing

    elif axis == 1:
        # Compute np.gradient along the second axis (axis=1)
        grad[:, 1:-1, :] = (arr[:, 2:, :] - arr[:, :-2, :]) / (2 * spacing)
        grad[:, 0, :] = (arr[:, 1, :] - arr[:, 0, :]) / spacing
        grad[:, -1, :] = (arr[:, -1, :] - arr[:, -2, :]) / spacing

    elif axis == 2:
        # Compute np.gradient along the third axis (axis=2)
        grad[:, :, 1:-1] = (arr[:, :, 2:] - arr[:, :, :-2]) / (2 * spacing)
        grad[:, :, 0] = (arr[:, :, 1] - arr[:, :, 0]) / spacing
        grad[:, :, -1] = (arr[:, :, -1] - arr[:, :, -2]) / spacing

    return grad
    return grad

def calculate_curl(F, dx, dy, dz):
    Fx, Fy, Fz = F[0], F[1], F[2]
    
    # Partial derivatives
    dFz_dy = np.gradient(Fz, dy, axis=1)
    dFy_dz = np.gradient(Fy, dz, axis=2)
    dFx_dz = np.gradient(Fx, dz, axis=2)
    dFz_dx = np.gradient(Fz, dx, axis=0)
    dFy_dx = np.gradient(Fy, dx, axis=0)
    dFx_dy = np.gradient(Fx, dy, axis=1)
    
    # Curl components
    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy
    
    # Combine components into a single array
    curl = np.array([curl_x, curl_y, curl_z])
    return curl


@numba.njit()
def cross_product(A, B):
    A_x, A_y, A_z = A[0], A[1], A[2]
    B_x, B_y, B_z = B[0], B[1], B[2]
    
    # Calculate each component of the cross product
    C_x = A_y * B_z - A_z * B_y
    C_y = A_z * B_x - A_x * B_z
    C_z = A_x * B_y - A_y * B_x
    
    # Combine components into a single array
    C = np.array([C_x, C_y, C_z])
    return C


# Particle Class Definition
class Particle:
    """
    Represents a charged particle with position, velocity, charge, and mass.
    """
    def __init__(self, position, velocity, charge, mass):
        self.position = np.array(position, dtype=float)  # Position vector [x, y, z]
        self.velocity = np.array(velocity, dtype=float)  # Velocity vector [vx, vy, vz]
        self.charge = charge  # Charge (Coulombs)
        self.mass = mass      # Mass (kilograms)


# @numba.njit(parallel=True)
@numba.njit
def compute_potentials(pos, q, v, xsize, ysize, zsize, dx, dy, dz):
    V = np.zeros((zsize, ysize, xsize))
    Ax = np.zeros((zsize, ysize, xsize))
    Ay = np.zeros((zsize, ysize, xsize))
    Az = np.zeros((zsize, ysize, xsize))
    
    epsilon = 1e-20
    r_cutoff = 1000 # change later
    
    # for z in numba.prange(zsize):
    #     for y in range(ysize):
    #         for x in range(xsize):
    #             for p in particles:
    #                 rx = x * dx - p["position"][0]
    #                 ry = y * dy - p["position"][1]
    #                 rz = z * dz - p["position"][2]
    #                 r_squared = rx**2 + ry**2 + rz**2
                    
    #                 if r_squared < r_cutoff**2:
    #                     r = np.sqrt(r_squared) + epsilon
    #                     V[z, y, x] += p["charge"] / (4 * np.pi * epsilon_0 * r)
    #                     Ax[z, y, x] += (p["charge"] * p["velocity"][0] * mu_0) / (4 * np.pi * r)
    #                     Ay[z, y, x] += (p["charge"] * p["velocity"][1] * mu_0) / (4 * np.pi * r)
    #                     Az[z, y, x] += (p["charge"] * p["velocity"][2] * mu_0) / (4 * np.pi * r)
    

    for z in numba.prange(zsize):
    # for z in range(zsize):
        for y in range(ysize):
            for x in range(xsize):
                rx = x * dx - pos[0]
                ry = y * dy - pos[1]
                rz = z * dz - pos[2]
                r_squared = rx**2 + ry**2 + rz**2
                
                # if r_squared:
                r = np.sqrt(r_squared) + epsilon
                V[z, y, x] += q / (4 * np.pi * epsilon_0 * r)
                Ax[z, y, x] += (q * v[0] * mu_0) / (4 * np.pi * r)
                Ay[z, y, x] += (q * v[1] * mu_0) / (4 * np.pi * r)
                Az[z, y, x] += (q * v[2] * mu_0) / (4 * np.pi * r)
    
    return V, Ax, Ay, Az




class Plane:
    def __init__(self, xsize, ysize, zsize, grid_spacing, dtype=np.float64):
        self.xsize = xsize
        self.ysize = ysize
        self.zsize = zsize
        self.dx = self.dy = self.dz = self.grid_spacing = grid_spacing
        self.dtype = dtype
        self.particles = []
        self.time = 0
        
        # Electric potential grid
        self.V = np.zeros((self.zsize, self.ysize, self.xsize),dtype=self.dtype)
        
        # Magnetic potential grid
        self.Ax = np.zeros((self.zsize, self.ysize, self.xsize),dtype=self.dtype)
        self.Ay = np.zeros((self.zsize, self.ysize, self.xsize),dtype=self.dtype)
        self.Az = np.zeros((self.zsize, self.ysize, self.xsize),dtype=self.dtype)
        
        # Electric field grid
        self.E = np.zeros((3, self.zsize, self.ysize, self.xsize))
        
        # Magnetic field grid
        self.B = np.zeros((3, self.zsize, self.ysize, self.xsize))
        
    def add_particle(self, particle: Particle):
        self.particles.append(particle)
        
    def update_fields(self, timestep):
        """
        Update the electric and magnetic fields using retarded potentials for causality.
        """
        # Create empty grids for the new potentials
        newV = np.zeros_like(self.V, dtype=np.float64)
        newAx = np.zeros_like(self.Ax, dtype=np.float64)
        newAy = np.zeros_like(self.Ay, dtype=np.float64)
        newAz = np.zeros_like(self.Az, dtype=np.float64)
        
        # Speed of light
        c = 3e8  # m/s
        
        # Loop over particles to calculate the retarded potentials
        for p in self.particles:
            # Compute the retarded potentials for each grid point
            for ix in range(self.xsize):
                for iy in range(self.ysize):
                    for iz in range(self.zsize):
                        # Position of the grid point
                        grid_position = np.array([ix * self.dx, iy * self.dy, iz * self.dz])
                        
                        # Compute the distance between the particle and the grid point
                        r_vec = grid_position - p.position
                        r = np.linalg.norm(r_vec)  # Magnitude of the distance
                        
                        # Compute the retarded time
                        t_retarded = self.time - r / c
                        
                        # Evaluate the particle's position and velocity at the retarded time
                        # Assuming linear motion for simplicity (modify for other motion):
                        p_retarded_position = p.position + p.velocity * (t_retarded - self.time)
                        v_retarded = p.velocity  # Velocity assumed constant
                        
                        # Recompute distance vector at the retarded position
                        r_vec = grid_position - p_retarded_position
                        r = np.linalg.norm(r_vec) + 1e-12  # Avoid division by zero
                        
                        # Compute the scalar and vector potentials
                        newV[ix, iy, iz] += p.charge / (4 * np.pi * 8.854e-12 * r)
                        newAx[ix, iy, iz] += p.charge * v_retarded[0] / (4 * np.pi * 8.854e-12 * c * r)
                        newAy[ix, iy, iz] += p.charge * v_retarded[1] / (4 * np.pi * 8.854e-12 * c * r)
                        newAz[ix, iy, iz] += p.charge * v_retarded[2] / (4 * np.pi * 8.854e-12 * c * r)
        
        # Update the fields
        dAdt = np.array([(newAx - self.Ax), (newAy - self.Ay), (newAz - self.Az)], dtype=np.float64) / timestep
        gradV = np.array(np.gradient(-newV, self.dx), dtype=np.float64)
        
        # Electric field: E = -grad(V) - dA/dt
        self.E[0] = gradV[0] - dAdt[0]
        self.E[1] = gradV[1] - dAdt[1]
        self.E[2] = gradV[2] - dAdt[2]
        
        # Magnetic field: B = curl(A)
        self.B = calculate_curl([newAx, newAy, newAz], self.dx, self.dy, self.dz)
        
        # Update stored potentials
        self.V = newV
        self.Ax = newAx
        self.Ay = newAy
        self.Az = newAz
        
    def radial_gradient(self, array, position):
        # Create a grid of indices
        x, y, z = np.indices(array.shape)
        x, y, z = x * self.dx, y * self.dy, z *self.dz
        
        # center = np.array(center, dtype=int)
        center = position
        # Calculate the distance from the center (x, y, z) for each point in the array
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        
        # Calculate the gradients along each axis
        grad_x, grad_y, grad_z = np.gradient(array)
        
        # Radial component of the gradient (directional gradient along the radial vector)
        grad_radial = (grad_x * (x - center[0]) + grad_y * (y - center[1]) + grad_z * (z - center[2])) / distances
        
        # Replace NaN values (for the center point) with zero (since gradient at the center is undefined)
        grad_radial[np.isnan(grad_radial)] = 0
        
        return grad_radial
    
    def spherical_gradient(self, array, position):
        x,y,z = position
        
        # Define the step size for numerical differentiation
        step = 1
        
        # Gradient in Cartesian coordinates
        grad_x = (array[x + step, y, z] - array[x - step, y, z]) / (2 * step)
        grad_y = (array[x, y + step, z] - array[x, y - step, z]) / (2 * step)
        grad_z = (array[x, y, z + step] - array[x, y, z - step]) / (2 * step)
        
        # Calculate the spherical gradient components
        r = np.sqrt(x**2 + y**2 + z**2)  # Radial distance from the origin
        theta = np.arccos(z / r)  # Polar angle
        phi = np.arctan2(y, x)  # Azimuthal angle
        
        # Gradient in spherical coordinates (radial, polar, azimuthal)
        grad_r = grad_x * (x / r) + grad_y * (y / r) + grad_z * (z / r)
        grad_theta = grad_x * (x * z) / (r**2 * np.sin(theta)) + grad_y * (y * z) / (r**2 * np.sin(theta)) - grad_z / r
        grad_phi = -grad_x * (y / r**2) + grad_y * (x / r**2)
        
        # Transform spherical gradient back to Cartesian coordinates
        grad_cartesian_x = grad_r * np.sin(theta) * np.cos(phi) + grad_theta * np.cos(theta) * np.cos(phi) - grad_phi * np.sin(phi)
        grad_cartesian_y = grad_r * np.sin(theta) * np.sin(phi) + grad_theta * np.cos(theta) * np.sin(phi) + grad_phi * np.cos(phi)
        grad_cartesian_z = grad_r * np.cos(theta) - grad_theta * np.sin(theta)
        
        print(grad_cartesian_x.shape)
        return np.array([grad_cartesian_x, grad_cartesian_y, grad_cartesian_z])
    
    def initialize_fields(self):
        # Create empty grids for new potentials
        newV = np.zeros_like(self.V,dtype=np.float64)
        newAx = np.zeros_like(self.Ax,dtype=np.float64)
        newAy = np.zeros_like(self.Ay,dtype=np.float64)
        newAz = np.zeros_like(self.Az,dtype=np.float64)
        
        for p in self.particles:
            out = compute_potentials(p.position, p.charge, p.velocity, self.xsize, self.ysize, self.zsize, 
                        self.dx, self.dy, self.dz)
            newV += out[0]
            
            # update field
            indices = p.position * np.array([1/self.dx, 1/self.dy, 1/self.dz])

            self.E += self.spherical_gradient(newV, np.array(indices, dtype=int))

        
        
        self.V = newV
        self.Ax = newAx
        self.Ay = newAy
        self.Az = newAz


class Integrator:
    def __init__(self, grid_size: tuple, grid_spacing=1e-2, path="output.xyz", plot_every=None):
        self.grid_spacing = grid_spacing
        self.grid_size = grid_size
        self.plane = Plane(*grid_size, self.grid_spacing)
        self.path = path
        self.plot_every = plot_every
        
    def add_particles(self, particles: list):
        for p in particles:
            self.plane.add_particle(p)
    
    def initialize_fields(self):
        self.plane.initialize_fields()
        

    def plot2dslice(self):
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        
        # Extract a 2D slice (e.g., at z=0) of the 3D electric field and potential
        z_slice = int(self.plane.zsize/2) -1
        # z_slice = 0
        Ex_2D = self.plane.E[0][z_slice, :, :]
        Ey_2D = self.plane.E[1][z_slice, :, :]
        V_2D = self.plane.V[z_slice, :, :]
        
        Bx_2D = self.plane.B[0][z_slice, :, :]
        By_2D = self.plane.B[1][z_slice, :, :]
        
        # Create a meshgrid for plotting
        y, x = np.arange(Ex_2D.shape[0]), np.arange(Ex_2D.shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Plot electric field as a streamplot
        axes[0].streamplot(X, Y, Ex_2D, Ey_2D, color=np.sqrt(Ex_2D**2 + Ey_2D**2), cmap='viridis')
        axes[0].set_title('Electric Field (E)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        
        # Plot magnetic field as a streamplot
        axes[1].streamplot(X, Y, Bx_2D, By_2D, color=np.sqrt(Bx_2D**2 + By_2D**2), cmap='viridis')
        axes[1].set_title('Magnetic Field (B)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        
        # Plot electric potential
        im = axes[2].imshow(V_2D, extent=[0, Ex_2D.shape[1], 0, Ex_2D.shape[0]], origin='lower', cmap='plasma')
        axes[2].set_title('Electric Potential (V)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        fig.colorbar(im, ax=axes[2], label='Potential (V)')
        
        # Plot particle positions
        axes[3].scatter([p.position[0] for p in self.plane.particles], [p.position[1] for p in self.plane.particles], c='red', label='Particles')
        print([p.position for p in self.plane.particles])
        # axes[2].invert_yaxis()
        axes[3].set_title('Particle Positions')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        axes[3].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot3dfield(self,n: int,showfig=True):
        """
        Plot two 3D vector fields side by side.

        Parameters:
        - vector_field1: A tuple (X, Y, Z, U, V, W) for the first field.
        - vector_field2: A tuple (X, Y, Z, U, V, W) for the second field.
        - title1: Title for the first plot.
        - title2: Title for the second plot.
        """
        fig = plt.figure(figsize=(14, 6))

        # Generate a 3D grid in proper order
        x, y, z = np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), np.arange(self.grid_size[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # Use 'ij' indexing for proper dimensional mapping

        # Plot first vector field
        ax1 = fig.add_subplot(121, projection='3d')
        U1, V1, W1 = self.plane.E
        ax1.quiver(
            X[::n, ::n, ::n], Y[::n, ::n, ::n], Z[::n, ::n, ::n], 
            U1[::n, ::n, ::n], V1[::n, ::n, ::n], W1[::n, ::n, ::n], 
            length=5, normalize=True
        )
        ax1.set_title("Electric Field (E)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        plt.tight_layout()
        if showfig:
            plt.show()
        
    def create_particle_shapes(self, shape, scale):
            
            if shape == "square":
                return
            
    def create_random_particles(self, n, q, m, xsize, ysize, zsize, max_velocity, min_velocity = 0) -> list:
        particles = []
        q, m = [q,], [m,]
        
        
        q, m = list(q), list(m)
        if n > 1 and (len(q) != n) and (len(q) != len(m)):
            raise IndexError(f"Lenghts of q and m ({len(q)}, {len(m)}) must match") 
        c = [-1,1]
        for i in range(n):
            x, y, z = np.random.uniform([0,0,0], np.array([xsize, ysize, zsize]) * self.grid_spacing, 3)
            vx, vy, vz = np.random.uniform(min_velocity, max_velocity, 3)
            p = Particle(
                position=[x,y,z],
                velocity=[vx,vy,vz],
                charge = q.pop() * random.choice(-1,1) if len(q) > 1 else q[0],
                mass = m.pop() if len(m) > 1 else m[0]
            )
            particles.append(p)
            
        self.add_particles(particles)
    
        
    def simulate(self, dt=1e-6, N_steps = 10, save_animation=False):
        if len(self.plane.particles) == 0:
            print("No particles initialized in simulation")
            return
        
        
        # Grid coordinates scaled to specified spacing
        x = np.arange(self.grid_size[0]) * self.grid_spacing
        y = np.arange(self.grid_size[1]) * self.grid_spacing
        z = np.arange(self.grid_size[2]) * self.grid_spacing
        
        def get_fields(position, E_interpolators: list, B_interpolators: list):
            pos = np.mod(position, [self.grid_size[0] * self.grid_spacing, self.grid_size[1] * self.grid_spacing, self.grid_size[2]] * self.grid_spacing)
            interp_pos = np.array([pos])
            Ex, Ey, Ez = [interp(interp_pos)[0] for interp in E_interpolators]
            Bx, By, Bz = [interp(interp_pos)[0] for interp in B_interpolators]
            return np.array([Ex, Ey, Ez]), np.array([Bx, By, Bz])

        def get_fields_nointerp(position,E,B):
            pos = np.mod(position, [self.grid_size[0] * self.grid_spacing, self.grid_size[1] * self.grid_spacing, self.grid_size[2] * self.grid_spacing])
            pos = np.array(pos/self.grid_spacing, dtype=int)
            Ex, Ey, Ez = np.array([E[0][pos[0],pos[1],pos[2]], E[1][pos[0],pos[1],pos[2]], E[2][pos[0],pos[1],pos[2]]])
            Bx, By, Bz = np.array([B[0][pos[0],pos[1],pos[2]], B[1][pos[0],pos[1],pos[2]], B[2][pos[0],pos[1],pos[2]]])
            return np.array([Ex, Ey, Ez]), np.array([Bx, By, Bz])

        def reflect(v_in: np.ndarray, n: np.ndarray, e):
            n = n / np.linalg.norm(n)
            v_n = np.dot(v_in, n) * n
            v_t = v_in - v_n
            v_out = v_t - e * v_n
            return v_out

        times = np.zeros(N_steps + 1)
        times[0] = 0.0
        
        if save_animation:
                # Setup the figure
            self.fig = plt.figure(figsize=(14, 6))
            x, y, z = np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), np.arange(self.grid_size[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            m = 10
            # First subplot for the electric field (E)
            self.ax1 = self.fig.add_subplot(111, projection='3d')
            U1, V1, W1 = self.plane.E
            self.quiver1 = self.ax1.quiver(
            X[::m, ::m, ::m], Y[::m, ::m, ::m], Z[::m, ::m, ::m], 
            U1[::m, ::m, ::m], V1[::m, ::m, ::m], W1[::m, ::m, ::m], 
            length=5, normalize=True
            )
            self.ax1.set_title("Electric Field (E)")
            self.ax1.set_xlabel("X")
            self.ax1.set_ylabel("Y")
            self.ax1.set_zlabel("Z")
            self.ax1.set_xlim(self.plane.xsize)
            self.ax1.set_ylim(self.plane.ysize)
            self.ax1.set_zlim(self.plane.zsize)
        
        def update():
            if self.quiver1 is not None:
                try:
                    self.quiver1.remove()
                except:
                    pass
            m = 10
            U1, V1, W1 = self.plane.E
            x, y, z = np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), np.arange(self.grid_size[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # self.quiver1 = self.ax1.quiver(
            # X[::m, ::m, ::m], Y[::m, ::m, ::m], Z[::m, ::m, ::m], 
            # U1[::m, ::m, ::m], V1[::m, ::m, ::m], W1[::m, ::m, ::m], 
            # length=5, normalize=True
            # )
            self.quiver11 = self.ax1.scatter([100],[100],[100],s=10)
            return self.quiver1,
        
        xyz_file = open(self.path, 'w')
        print("Starting simulation")
        
        def loop(n):
            
            if save_animation:
                return update()
            
            if self.plot_every is not None and n % int(self.plot_every) == 0:
                self.plot2dslice()
                # self.plot3dfield(10) # run if you want to crash your pc
            
            
            times[n + 1] = times[n] + dt
            self.plane.time = times[n]
            
            print(self.plane.particles[0].position)
            
            # Write header for this iteration
            # xyz_file = open(self.path, 'w')
            xyz_file.write(f'{len(self.plane.particles)}\n')
            xyz_file.write(f'Time={times[n + 1]:.6e} s\n')
            # xyz_file.close()
            
            self.average_velocity = np.mean([np.linalg.norm(p.velocity) for p in self.plane.particles])
            # Get fields for each particle
            for p in self.plane.particles:
                # E, B = get_fields(p.position, [Ex_interp, Ey_interp, Ez_interp], [Bx_interp, By_interp, Bz_interp])
                E,B = get_fields_nointerp(p.position,self.plane.E,self.plane.B)
                
                # Calculate acceleration of particle
                acceleration = (p.charge / p.mass) * (E + cross_product(p.velocity, B))
                # acceleration = (p.charge / p.mass) * (E)
                
                # Update particle using Euler's method
                p.velocity += acceleration * dt
                p.position += p.velocity * dt
                p.position[p.position == 0] = self.grid_spacing
                
                
                def apply_boundary(box_length, type="periodic"):
                    if type == "periodic":
                        for p in self.plane.particles:
                            p.position = np.mod(p.position, [self.plane.xsize, self.plane.ysize, self.plane.zsize])
                            p.position[p.position == 0] = self.grid_spacing
                            # for i in range(len(p.position)):  # Loop over x, y, z components
                                # Wrap the position to stay within [0, L)
                                # if p.position[i] < 0:
                                #     p.position[i] = box_length - 1e-9
                                # elif p.position[i] >= box_length:
                                #     p.position[i] = 1e-9
                    if type == "none":
                        pass
                            
                
                apply_boundary(self.plane.xsize,"none")
                
                
                # plt.scatter([p.position[0] for p in self.plane.particles], [p.position[1] for p in self.plane.particles])
                # plt.show()
                
                # Write particle position
                x_pos, y_pos, z_pos = p.position
                # xyz_file = open(self.path, 'w')
                xyz_file.write(f'P {x_pos:.6e} {y_pos:.6e} {z_pos:.6e}\n')
                # xyz_file.close()
                
                print(f"Step: {n}, average velocity: {self.average_velocity}", end="\r")
            
            # Update fields
            self.plane.update_fields(timestep=dt)
            
            
                
        
        if save_animation:
            self.anim = FuncAnimation(self.fig, loop, frames=N_steps, blit=False)
            writer = FFMpegWriter(fps=1)
            self.anim.save('animation.mp4', writer=writer)
        else:
        # Time-stepping loop
            for n in range(N_steps):
                loop(n)
            
        xyz_file.close()
        print("\nSimulation complete")


if __name__ == "__main__":
    
    plot_every = None
    if len(sys.argv) > 1:
        plot_every = int(sys.argv[1])
    
    integrator = Integrator((50,50,50),grid_spacing=1e-25, plot_every=plot_every)
    # integrator = Integrator((50,50,50), grid_spacing=1, plot_every=plot_every)
    
    # Particle Initialization
    a = Particle(
        position=[150, 150, 199],         
        velocity=[0.0, 0.0, 0.0],         # Initially at rest
        charge=elementary_charge,           # Electron charge
        mass=electron_mass               # Electron mass
    )
    
    # Particle Initialization
    b = Particle(
        position=[(integrator.plane.xsize + 10) * integrator.plane.grid_spacing, integrator.plane.xsize * integrator.plane.grid_spacing, integrator.plane.xsize * integrator.plane.grid_spacing],
        velocity=[0.0, 0.0, 0.0],
        charge=-elementary_charge*10,
        mass=electron_mass
    )
    
    c = Particle(
        position=[integrator.plane.xsize * integrator.plane.grid_spacing * 1/2, integrator.plane.xsize * integrator.plane.grid_spacing * 1/2, integrator.plane.xsize * integrator.plane.grid_spacing * 0.6],
        velocity=[0.0, 0.0, 0.0],
        charge=elementary_charge*10,
        mass=electron_mass
    )
    
    
    
    # integrator.create_random_particles(50,elementary_charge * 1e5,electron_mass, integrator.plane.xsize, integrator.plane.ysize, integrator.plane.zsize, 0, 0)
    
    integrator.add_particles([c])
    
    # print(integrator.plane.particles[0].position)
    
    integrator.initialize_fields()
    integrator.simulate(dt=1e-20,N_steps=1000,save_animation=False)
    
    

