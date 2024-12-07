import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numba
from copy import deepcopy
import sys
from mpl_toolkits.mplot3d import Axes3D

# Constants
epsilon_0 = 8.854187817e-12  # Vacuum permittivity in SI units (F/m)
mu_0 = 1.25663706127e-6  # Vacuum permeability in SI units (N/A**2)
elementary_charge = 1.602e-19
electron_mass = 9.109e-31

# testing
# epsilon_0 = 1
# mu_0 = 1
elementary_charge = 1
electron_mass = 1e-5

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


@numba.njit(parallel=True)
def compute_potentials(particles, xsize, ysize, zsize, dx, dy, dz):
    V = np.zeros((zsize, ysize, xsize))
    Ax = np.zeros((zsize, ysize, xsize))
    Ay = np.zeros((zsize, ysize, xsize))
    Az = np.zeros((zsize, ysize, xsize))
    
    epsilon = 1e-10
    r_cutoff = 1000 # change later
    
    for z in numba.prange(zsize):
        for y in range(ysize):
            for x in range(xsize):
                for p in particles:
                    rx = x * dx - p["position"][0]
                    ry = y * dy - p["position"][1]
                    rz = z * dz - p["position"][2]
                    r_squared = rx**2 + ry**2 + rz**2
                    
                    if r_squared < r_cutoff**2:
                        r = np.sqrt(r_squared) + epsilon
                        V[z, y, x] += p["charge"] / (4 * np.pi * epsilon_0 * r)
                        Ax[z, y, x] += (p["charge"] * p["velocity"][0] * mu_0) / (4 * np.pi * r)
                        Ay[z, y, x] += (p["charge"] * p["velocity"][1] * mu_0) / (4 * np.pi * r)
                        Az[z, y, x] += (p["charge"] * p["velocity"][2] * mu_0) / (4 * np.pi * r)
    
    return V, Ax, Ay, Az




class Plane:
    def __init__(self, xsize, ysize, zsize, grid_spacing):
        self.xsize = xsize
        self.ysize = ysize
        self.zsize = zsize
        self.dx = self.dy = self.dz = grid_spacing
        self.particles = []
        
        # Electric potential grid
        self.V = np.zeros((self.zsize, self.ysize, self.xsize))
        
        # Magnetic potential grid
        self.Ax = np.zeros((self.zsize, self.ysize, self.xsize))
        self.Ay = np.zeros((self.zsize, self.ysize, self.xsize))
        self.Az = np.zeros((self.zsize, self.ysize, self.xsize))
        
        # Electric field grid
        self.E = np.zeros((3, self.zsize, self.ysize, self.xsize))
        
        # Magnetic field grid
        self.B = np.zeros((3, self.zsize, self.ysize, self.xsize))
        
    def add_particle(self, particle: Particle):
        self.particles.append(particle)
        
    def update_fields(self, timestep):
        # Create empty grids for new potentials
        newV = np.zeros_like(self.V)
        newAx = np.zeros_like(self.Ax)
        newAy = np.zeros_like(self.Ay)
        newAz = np.zeros_like(self.Az)
        
        # Convert particle data into a structured array for Numba
        particle_array = np.array([(p.charge, p.position, p.velocity) for p in self.particles],
                    dtype=[('charge', np.float64), ('position', np.float64, 3), ('velocity', np.float64, 3)])
        
        # Compute potentials using Numba-accelerated function
        newV, newAx, newAy, newAz = compute_potentials(particle_array, self.xsize, self.ysize, self.zsize, 
                        self.dx, self.dy, self.dz)
        
        # Update plane fields
        dAdt = np.array([newAx - self.Ax, newAy - self.Ay, newAz - self.Az]) / timestep
        self.E[0], self.E[1], self.E[2] = np.gradient(-newV, self.dx) - dAdt
        # self.E[0], self.E[1], self.E[2] = np.gradient(-newV, self.dx)
        self.B = calculate_curl([newAx, newAy, newAz], self.dx, self.dy, self.dz)
        
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
        self.plane.update_fields(timestep=1e-3)
        

    def plot2dslice(self):
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        
        # Extract a 2D slice (e.g., at z=0) of the 3D electric field and potential
        z_slice = 50
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
        # axes[2].invert_yaxis()
        axes[3].set_title('Particle Positions')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        axes[3].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot3dfield(self):
        """
        Plot two 3D vector fields side by side.

        Parameters:
        - vector_field1: A tuple (X, Y, Z, U, V, W) for the first field.
        - vector_field2: A tuple (X, Y, Z, U, V, W) for the second field.
        - title1: Title for the first plot.
        - title2: Title for the second plot.
        """
        fig = plt.figure(figsize=(14, 6))
        
        x, y, z = np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), np.arange(self.grid_size[2])
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Plot first vector field
        ax1 = fig.add_subplot(121, projection='3d')
        U1, V1, W1 = self.plane.E
        ax1.quiver(X, Y, Z, U1, V1, W1, length=0.1, normalize=True)
        ax1.set_title("Electric Field (E)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        
        # Plot second vector field
        ax2 = fig.add_subplot(122, projection='3d')
        U2, V2, W2 = self.plane.B
        ax2.quiver(X, Y, Z, U2, V2, W2, length=0.1, normalize=True)
        ax2.set_title("Magnetic Field (B)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        
        plt.tight_layout()
        plt.show()
        
    def simulate(self, dt=1e-6, N_steps = 10):
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
            pos = np.mod(position, [self.grid_size[0] * self.grid_spacing, self.grid_size[1] * self.grid_spacing, self.grid_size[2]] * self.grid_spacing)
            pos = np.array(pos, dtype=int)
            Ex, Ey, Ez = np.array([E[0][pos[0],pos[1],pos[2]], E[1][pos[0],pos[1],pos[2]], E[2][pos[0],pos[1],pos[2]]])
            Bx, By, Bz = np.array([B[0][pos[0],pos[1],pos[2]], B[1][pos[0],pos[1],pos[2]], B[2][pos[0],pos[1],pos[2]]])
            return np.array([Ex, Ey, Ez]), np.array([Bx, By, Bz])


        times = np.zeros(N_steps + 1)
        times[0] = 0.0
        
        xyz_file = open(self.path, 'w')
        print("Starting simulation")
        # Time-stepping loop
        for n in range(N_steps):
            times[n + 1] = times[n] + dt
            
            # Update interpolators with new fields
            Ex, Ey, Ez = self.plane.E[0], self.plane.E[1], self.plane.E[2]
            Bx, By, Bz = self.plane.B[0], self.plane.B[1], self.plane.B[2]
            
            Ex_interp = RegularGridInterpolator((x, y, z), Ex)
            Ey_interp = RegularGridInterpolator((x, y, z), Ey)
            Ez_interp = RegularGridInterpolator((x, y, z), Ez)
            
            Bx_interp = RegularGridInterpolator((x, y, z), Bx)
            By_interp = RegularGridInterpolator((x, y, z), By)
            Bz_interp = RegularGridInterpolator((x, y, z), Bz)
            
            # Write header for this iteration
            # xyz_file = open(self.path, 'w')
            xyz_file.write(f'{len(self.plane.particles)}\n')
            xyz_file.write(f'Time={times[n + 1]:.5e} s\n')
            # xyz_file.close()
            
            self.average_velocity = np.mean([np.linalg.norm(p.velocity) for p in self.plane.particles])
            # Get fields for each particle
            for p in self.plane.particles:
                # E, B = get_fields(p.position, [Ex_interp, Ey_interp, Ez_interp], [Bx_interp, By_interp, Bz_interp])
                E,B = get_fields_nointerp(p.position,self.plane.E,self.plane.B)
                
                # Calculate acceleration of particle
                # acceleration = (p.charge / p.mass) * (E + cross_product(p.velocity, B))
                acceleration = (p.charge / p.mass) * (E)
                
                # Update particle using Euler's method
                p.velocity += acceleration * dt
                p.position += p.velocity * dt
                
                # periodic boundary
                p.position = np.mod(p.position, [self.grid_size[0] * self.grid_spacing, self.grid_size[1] * self.grid_spacing, self.grid_size[2]] * self.grid_spacing)
                
                # plt.scatter([p.position[0] for p in self.plane.particles], [p.position[1] for p in self.plane.particles])
                # plt.show()
                
                # Write particle position
                x_pos, y_pos, z_pos = p.position
                # xyz_file = open(self.path, 'w')
                xyz_file.write(f'P {x_pos:.6f} {y_pos:.6f} {z_pos:.6f}\n')
                # xyz_file.close()
                
                print(f"Iteration: {n}, average velocity: {self.average_velocity}", end="\r")
            
            # Update fields
            self.plane.update_fields(timestep=dt)
            
            
            if self.plot_every is not None and n % int(self.plot_every) == 0:
                self.plot2dslice()
                # self.plot3dfield() # run if you want to crash your pc
                
        
        xyz_file.close()
        print("Simulation complete")


if __name__ == "__main__":
    
    plot_every = None
    if len(sys.argv) > 1:
        plot_every = sys.argv[1]
    
    integrator = Integrator((200,200,200),grid_spacing=1, plot_every=plot_every)
    
    # Particle Initialization
    a = Particle(
        position=[150, 150, 199],         
        velocity=[0.0, 0.0, 0.0],         # Initially at rest
        charge=elementary_charge,           # Electron charge
        mass=electron_mass               # Electron mass
    )
    
    # Particle Initialization
    b = Particle(
        position=[100, 100, 150],
        velocity=[0.0, 0.0, 0.0],
        charge=-elementary_charge,
        mass=electron_mass
    )
    
    c = Particle(
        position=[100, 100, 50],
        velocity=[0.0, 0.0, 0.0],
        charge=elementary_charge,
        mass=electron_mass
    )
    
    particles = [a,b]
    
    integrator.add_particles(particles)
    
    integrator.initialize_fields()
    integrator.simulate(dt=0.5e-6,N_steps=400)
    
    

