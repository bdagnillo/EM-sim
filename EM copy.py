import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numba

# Constants
epsilon_0 = 8.854187817e-12  # Vacuum permittivity in SI units (F/m)
mu_0 = 1.25663706127e-6 #Vacuum permeability in SI units (N/A**2)
elementary_charge = 1.602e-19
electron_mass = 9.109e-31

@numba.njit(parallel=True)
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
        
        
class Plane:
    
    def __init__(self, xsize, ysize, zsize, grid_spacing):
        self.xsize = xsize
        self.ysize = ysize
        self.zsize = zsize
        self.dx = self.dy = self.dz = grid_spacing
        self.particles = []
        
        # electric potential grid
        self.V = np.zeros((self.ysize,self.xsize,self.zsize))
        
        # magnetic potential grid
        self.Ax = np.zeros((self.ysize,self.xsize,self.zsize)) 
        self.Ay = np.zeros((self.ysize,self.xsize,self.zsize))
        self.Az = np.zeros((self.ysize,self.xsize,self.zsize))
        
        # electric field grid
        self.E = np.zeros((self.ysize,self.xsize,self.zsize))
        
        # magnetic field grid
        self.B = np.zeros((self.ysize,self.xsize,self.zsize))
        
    def add_particle(self,particle: Particle):
        self.particles.append(particle)
        
    @numba.njit(parallel=True)
    def update_fields(self, timestep):
        
        # electric potential
        newV = np.zeros((self.ysize,self.xsize,self.zsize))
        
        # magnetic potential
        newAx = newAy = newAz = np.zeros((self.ysize,self.xsize,self.zsize))
        
        for z in range(self.zsize):
            for y in range(self.ysize):
                for x in range(self.xsize):
                    for p in self.particles:
                        rx = x - p.position[0]
                        ry = y - p.position[1]
                        rz = z - p.position[2]
                        r = np.sqrt(rx**2 + ry**2 + rz**2)
                        r[r==0] = 0.5/6 #fractional distance instead of 0, avoids div by zero and smooths out potential
                        newV[z,y,x] += p.charge / (4 * np.pi * epsilon_0* r)
                        
                        newAx[z,y,x] += (p.charge * p.velocity[0] * mu_0) / (4 * np.pi * r)
                        newAy[z,y,x] += (p.charge * p.velocity[1] * mu_0) / (4 * np.pi * r)
                        newAz[z,y,x] += (p.charge * p.velocity[2] * mu_0) / (4 * np.pi * r)
        
        # update plane fields
        dAdt = np.array([newAx - self.Ax, newAy - self.Ay, newAz - self.Az])/timestep
        self.E = np.gradient(-self.V, self.dx, self.dy, self.dz) - dAdt

        self.B = calculate_curl([newAx,newAy,newAz])
        
        self.V = newV
        self.Ax = newAx
        self.Ay = newAy
        self.Az = newAz
    

class Integrator:
    
    def __init__(self, grid_size: tuple, grid_spacing,path):
        self.grid_spacing = grid_spacing
        self.grid_size = grid_size
        self.plane = Plane(*grid_size, self.grid_spacing)
        self.path = path
        
    def add_particles(self, particles: list):
        for p in particles:
            self.plane.add_particle(p)
    
    def initialize_fields(self):
        self.plane.update_fields(timestep=1e-6)
        
    def simulate(self, dt=1e-11, duration=1e-6):
        
        if len(self.plane.particles) == 0:
            print("No particles initialized in simulation")
            return
        
        N_steps = int(T/dt)
        
        # Grid coordinates scaled to specified spacing
        x = np.arange(grid_size[0])
        y = np.arange(grid_size[1])
        z = np.arange(grid_size[2])

        
        def get_fields(position, E_interpolators, Bx_interp, By_interp):
            pos = np.array([position[0] % self.grid_size[0], position[1] % self.grid_size[1]])
            interp_pos = np.array([pos])
            Ex, Ey, Ez = [interp(pos)[0] for interp in E_interpolators]
            
            return np.array([Ex, Ey, Ez]), np.array([Bx,By,Bz])
        
        # Time-stepping loop
        for n in range(N_steps):
            
            # update interpolators with new fields
            Ex_interp = RegularGridInterpolator((x,y), plane.Ex.T)
            Ey_interp = RegularGridInterpolator((x,y), plane.Ex.T)
        
            Bx_interp = RegularGridInterpolator((x,y), plane.Bx.T)
            By_interp = RegularGridInterpolator((x,y), plane.By.T)
            
            time = n * dt
            # get fields for each particle
            for p in self.plane.particles:
                E,B = get_fields(p.position)
                
                # calculate force on particle
                F = particle.charge * (E + )
                
                # update particle using Euler's method
                
                particle.velocity += (particle.charge)




# Simulation Parameters
dx = dy = 1e-2  # Grid spacing in meters (1 cm)
grid_size = (100, 100)  # Grid size (1 m x 1 m)

# Initialize plane
plane = Plane(*grid_size)
potential, Ex_grid, Ey_grid = compute_potential_and_field(plane)

# Coordinates for the grid
x = np.arange(grid_size[1]) * dx  # x-coordinates of grid points
y = np.arange(grid_size[0]) * dy  # y-coordinates of grid points

# Create interpolators for the electric field components
Ex_interp = RegularGridInterpolator((x, y), Ex_grid.T)
Ey_interp = RegularGridInterpolator((x, y), Ey_grid.T)

def get_electric_field(position):
    pos = np.array(position)
    if (0 <= pos[0] <= x[-1]) and (0 <= pos[1] <= y[-1]):
        interp_pos = np.array([pos])
        Ex = Ex_interp(interp_pos)[0]
        Ey = Ey_interp(interp_pos)[0]
        return np.array([Ex, Ey])
    else:
        return np.array([0.0, 0.0])

# Particle Initialization
particle = Particle(
    position=[0.5, 0.3],         # Start near bottom center
    velocity=[0.0, 0.0],         # Initially at rest
    charge=-1.602e-19,           # Electron charge
    mass=9.109e-31               # Electron mass
)

# Simulation Parameters
dt = 1e-11      # Time step (10 picoseconds)
T = 1e-6        # Total simulation time (100 nanoseconds)
N_steps = int(T / dt)


# Arrays to store positions and velocities for visualization
positions = np.zeros((N_steps+1, 2))
velocities = np.zeros((N_steps+1, 2))
times = np.zeros(N_steps+1)

xyz_filename = 'particle_trajectory.xyz'
xyz_file = open(xyz_filename, 'w')

# Store initial conditions
positions[0] = particle.position
velocities[0] = particle.velocity
times[0] = 0.0

# Time-stepping loop
for n in range(N_steps):
    time = n * dt
    # Get electric field at particle's position
    E = get_electric_field(particle.position)
    # Update particle using simple Euler method
    particle.velocity += (particle.charge / particle.mass) * E * dt
    particle.position += particle.velocity * dt

    # Store data
    positions[n+1] = particle.position
    velocities[n+1] = particle.velocity
    times[n+1] = times[n] + dt

    # Write data to XYZ file
    # For single particle, number of atoms is 1
    xyz_file.write('1\n')
    xyz_file.write(f'Time={times[n+1]:.5e} s\n')
    # Assuming particle is represented as 'P' (could be any symbol)
    x_pos, y_pos = particle.position
    z_pos = 0.0  # Since it's a 2D simulation
    xyz_file.write(f'P {x_pos:.6e} {y_pos:.6e} {z_pos:.6e}\n')

# Close the XYZ file
xyz_file.close()

# Visualization of particle trajectory
plt.figure(figsize=(8, 6))
plt.imshow(potential.T, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
plt.colorbar(label='Electric Potential (V)')
plt.plot(positions[:,0], positions[:,1], color='red', linewidth=2, label='Electron Trajectory')
plt.scatter(positions[0,0], positions[0,1], color='white', label='Start')
plt.scatter(positions[-1,0], positions[-1,1], color='black', label='End')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Electron Trajectory in Electric Field')
plt.legend()
plt.show()