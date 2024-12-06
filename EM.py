import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Constants
epsilon_0 = 8.854187817e-12  # Vacuum permittivity in SI units (F/m)

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

# Grid and Charge Distribution
def create_charge_grid(grid_size=(100, 100)):
    plane = np.zeros(grid_size)
    # Place charges at specific grid indices
    plane[50, 50] = -1e-9    # Positive charge at center (nanoCoulombs)
    plane[20, 20] = -1e-9    # Negative charge
    plane[80, 80] = -1e-9    # Positive charge
    return plane

def compute_potential_and_field(plane):
    grid_shape = plane.shape
    x_indices, y_indices = np.indices(grid_shape)
    potential = np.zeros(grid_shape)

    # Coordinates of grid points
    x_coords = x_indices * dx
    y_coords = y_indices * dy

    # Compute potential due to each charge
    charges = []
    positions = []
    for y in range(grid_shape[0]):
        for x in range(grid_shape[1]):
            q = plane[y, x]
            if q != 0:
                charges.append(q)
                positions.append((x_coords[y, x], y_coords[y, x]))
    
    positions = np.array(positions)
    charges = np.array(charges)
    
    # Compute potential at each grid point due to all charges
    for idx, (q, pos) in enumerate(zip(charges, positions)):
        rx = x_coords - pos[0]
        ry = y_coords - pos[1]
        r = np.sqrt(rx**2 + ry**2)
        r[r == 0] = np.nan  # Avoid division by zero
        potential += q / (4 * np.pi * epsilon_0 * r)
    
    potential = np.nan_to_num(potential, nan=0.0)
    
    # Compute electric field components (negative gradient of potential)
    Ex_grid, Ey_grid = np.gradient(-potential, dx, dy)
    
    return potential, Ex_grid, Ey_grid

# Simulation Parameters
dx = dy = 1e-2  # Grid spacing in meters (1 cm)
grid_size = (100, 100)  # Grid size (1 m x 1 m)

# Create charge grid and compute potential and field
plane = create_charge_grid(grid_size)
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