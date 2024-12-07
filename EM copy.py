import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numba

# Constants
epsilon_0 = 8.854187817e-12  # Vacuum permittivity in SI units (F/m)
mu_0 = 1.25663706127e-6 #Vacuum permeability in SI units (N/A**2)
elementary_charge = 1.602e-19
electron_mass = 9.109e-31

# Particle Class Definition
class Particle:
    """
    Represents a charged particle with position, velocity, charge, and mass.
    """
    def __init__(self, position, velocity, charge, mass):
        self.position = np.array(position, dtype=float)  # Position vector [x, y]
        self.velocity = np.array(velocity, dtype=float)  # Velocity vector [vx, vy]
        self.charge = charge  # Charge (Coulombs)
        self.mass = mass      # Mass (kilograms)
        
        
class Plane:
    
    def __init__(self, xsize, ysize, grid_spacing):
        self.xsize = xsize
        self.ysize = ysize
        self.dx = self.dy = grid_spacing
        self.particles = []
        
        # electric potential grid
        self.V = np.zeros(self.ysize,self.xsize)
        
        # magnetic potential grid
        self.Ax = np.zeros(self.ysize,self.xsize) 
        self.Ay = np.zeros(self.ysize,self.xsize)
        
        # electric field grid
        self.Ex = np.zeros(self.ysize,self.xsize)
        self.Ey = np.zeros(self.ysize,self.xsize)
        
        # magnetic field grid
        self.Bx = np.zeros(self.ysize,self.xsize)
        self.By = np.zeros(self.ysize,self.xsize)
        
    def add_particle(self,particle: Particle):
        self.particles.append(particle)
        
    @numba.njit(parallel=True)
    def update_fields(self, timestep):
        
            
        
        # electric potential
        newV = np.zeros(self.ysize, self.xsize)
        
        # magnetic potential
        newAx = newAy = np.zeros(self.ysize,self.xsize)
        
        for y in range(self.ysize):
            for x in range(self.xsize):
                for p in self.particles:
                    rx = x - p.position[0]
                    ry = x - p.position[1]
                    r = np.sqrt(rx**2 + ry**2)
                    r[r==0] = 0.5/4 #fractional distance instead of 0, avoids div by zero and smooths out potential
                    newV[y,x] += p.charge / (4 * np.pi * epsilon_0* r)
                    
                    newAx[y,x] += (p.charge * p.velocity[0] * mu_0) / (4 * np.pi * r)
                    newAy[y,x] += (p.charge * p.velocity[0] * mu_0) / (4 * np.pi * r)
                    
        
        # update plane fields
        dAdt = np.array([newAx - self.Ax, newAy - self.Ay])/timestep
        self.E = np.gradient(-self.V, dx, dy) - dAdt

        self.B = np.array([newAy - self.Ay])/dx - np.array([newAx - self.Ax])/dy
        
        self.V = newV
        self.Ax = newAx
        self.Ay = newAy
    

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
        
        # Interpolators
        Ex_interp = RegularGridInterpolator((x,y), plane.Ex.T)
        Ey_interp = RegularGridInterpolator((x,y), plane.Ex.T)
        
        Bx_interp = RegularGridInterpolator((x,y), plane.Bx.T)
        By_interp = RegularGridInterpolator((x,y), plane.By.T)
        
        def get_fields(position):
            pos = np.array([position[0] % self.grid_size[0], position[1] % self.grid_size[1]])
            interp_pos = np.array([pos])
            Ex = Ex_interp(pos)[0]
            Ey = Ey_interp(pos)[0]
            Bx = Bx_interp(pos)[0]
            By = By_interp(pos)[0]
            return np.array([Ex, Ey]), np.array([Bx,By])
        
        def 




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