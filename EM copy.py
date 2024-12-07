import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numba

# Constants
epsilon_0 = 8.854187817e-12  # Vacuum permittivity in SI units (F/m)
mu_0 = 1.25663706127e-6  # Vacuum permeability in SI units (N/A**2)
elementary_charge = 1.602e-19
electron_mass = 9.109e-31

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
def compute_potentials(particles, xsize, ysize, zsize, dx, dy, dz, V, Ax, Ay, Az):
    for z in numba.prange(zsize):
        for y in range(ysize):
            for x in range(xsize):
                for p in particles:
                    rx = x * dx - p["position"][0]
                    ry = y * dy - p["position"][1]
                    rz = z * dz - p["position"][2]
                    r = np.sqrt(rx**2 + ry**2 + rz**2)
                    if r == 0:
                        r = 0.5 / 6  # Fractional distance instead of 0

                    V[z, y, x] += p["charge"] / (4 * np.pi * epsilon_0 * r)
                    Ax[z, y, x] += (p["charge"] * p["velocity"][0] * mu_0) / (4 * np.pi * r)
                    Ay[z, y, x] += (p["charge"] * p["velocity"][1] * mu_0) / (4 * np.pi * r)
                    Az[z, y, x] += (p["charge"] * p["velocity"][2] * mu_0) / (4 * np.pi * r)

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
        
    def add_particle(self, particle):
        self.particles.append(particle)
        
    def update_fields(self, timestep):
        # Create empty grids for new potentials
        newV = np.zeros_like(self.V)
        newAx = np.zeros_like(self.Ax)
        newAy = np.zeros_like(self.Ay)
        newAz = np.zeros_like(self.Az)
        
        # Convert particle data into a structured array for Numba
        particle_array = np.array([
            (p.charge, p.position, p.velocity)
            for p in self.particles
        ], dtype=[('charge', np.float64), ('position', np.float64, 3), ('velocity', np.float64, 3)])
        
        # Compute potentials using Numba-accelerated function
        compute_potentials(particle_array, self.xsize, self.ysize, self.zsize, 
                        self.dx, self.dy, self.dz, newV, newAx, newAy, newAz)
        
        # Update plane fields
        dAdt = np.array([newAx - self.Ax, newAy - self.Ay, newAz - self.Az]) / timestep
        self.E[0], self.E[1], self.E[2] = np.gradient(-self.V, self.dx, self.dy, self.dz) - dAdt
        self.B = calculate_curl([newAx, newAy, newAz], self.dx, self.dy, self.dz)
        
        self.V = newV
        self.Ax = newAx
        self.Ay = newAy
        self.Az = newAz


class Integrator:
    def __init__(self, grid_size: tuple, grid_spacing=1e-2, path="output.xyz"):
        self.grid_spacing = grid_spacing
        self.grid_size = grid_size
        self.plane = Plane(*grid_size, self.grid_spacing)
        self.path = path
        
    def add_particles(self, particles: list):
        for p in particles:
            self.plane.add_particle(p)
    
    def initialize_fields(self):
        self.plane.update_fields(timestep=1e-6)
        
    def simulate(self, dt=1e-11, N_steps = 10):
        if len(self.plane.particles) == 0:
            print("No particles initialized in simulation")
            return
        
        
        # Grid coordinates scaled to specified spacing
        x = np.arange(self.grid_size[0])
        y = np.arange(self.grid_size[1])
        z = np.arange(self.grid_size[2])
        
        def get_fields(position, E_interpolators: list, B_interpolators: list):
            pos = np.mod(position, [self.grid_size[0], self.grid_size[1], self.grid_size[2]])
            interp_pos = np.array([pos])
            Ex, Ey, Ez = [interp(interp_pos)[0] for interp in E_interpolators]
            Bx, By, Bz = [interp(interp_pos)[0] for interp in B_interpolators]
            return np.array([Ex, Ey, Ez]), np.array([Bx, By, Bz])
        
        xyz_file = open(self.path, 'w')
        times = np.zeros(N_steps + 1)
        times[0] = 0.0
        
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
            xyz_file.write(f'{len(self.plane.particles)}\n')
            xyz_file.write(f'Time={times[n + 1]:.5e} s\n')
            
            # Get fields for each particle
            for p in self.plane.particles:
                E, B = get_fields(p.position, [Ex_interp, Ey_interp, Ez_interp], [Bx_interp, By_interp, Bz_interp])
                
                # Calculate acceleration of particle
                acceleration = (p.charge / p.mass) * (E + cross_product(p.velocity, B))
                
                # Update particle using Euler's method
                p.velocity += acceleration * dt
                p.position += p.velocity * dt
                
                # Write particle position
                x_pos, y_pos, z_pos = p.position
                xyz_file.write(f'P {x_pos:.3f} {y_pos:.3f} {z_pos:.3f}\n')
            
            # Update fields
            self.plane.update_fields(timestep=dt)
        
        xyz_file.close()


if __name__ == "__main__":
    
    integrator = Integrator((100,100,100))
    
    # Particle Initialization
    a = Particle(
        position=[0.5, 0.3, 0.3],         # Start near bottom center
        velocity=[0.0, 0.0, 0.0],         # Initially at rest
        charge=-1.602e-19,           # Electron charge
        mass=9.109e-31               # Electron mass
    )
    
    # Particle Initialization
    b = Particle(
        position=[0.8, 0.2, 0.2],
        velocity=[0.0, 0.0, 0.0],
        charge=-1.602e-19,
        mass=9.109e-31
    )
    
    particles = [a,b]
    
    integrator.add_particles(particles)
    
    integrator.simulate()
    
    

