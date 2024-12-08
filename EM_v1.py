import numpy as np
import sys

# Physical constants
epsilon_0 = 8.854187817e-12
mu_0 = 1.2566370614e-6
c = 1/np.sqrt(mu_0*epsilon_0)

# Particle class
class Particle:
    def __init__(self, position, velocity, charge, mass):
        self.position = np.array(position, dtype=float)  # [x, y, z]
        self.velocity = np.array(velocity, dtype=float)  # [vx, vy, vz]
        self.charge = charge
        self.mass = mass

def deposit_charge_current_3d(particles, rho, J, dx, dt):
    nx, ny, nz = rho.shape
    rho[...] = 0.0
    J[...] = 0.0
    for p in particles:
        x_cell = p.position[0]/dx
        y_cell = p.position[1]/dx
        z_cell = p.position[2]/dx
        
        i0 = int(np.floor(x_cell)) % nx
        j0 = int(np.floor(y_cell)) % ny
        k0 = int(np.floor(z_cell)) % nz
        
        fx = x_cell - i0
        fy = y_cell - j0
        fz = z_cell - k0
        
        # Weights
        wx0 = 1 - fx
        wx1 = fx
        wy0 = 1 - fy
        wy1 = fy
        wz0 = 1 - fz
        wz1 = fz

        i1 = (i0+1)%nx
        j1 = (j0+1)%ny
        k1 = (k0+1)%nz

        q = p.charge
        vx, vy, vz = p.velocity
        for (ix, wx) in [(i0, wx0), (i1, wx1)]:
            for (iy, wy) in [(j0, wy0), (j1, wy1)]:
                for (iz, wz) in [(k0, wz0), (k1, wz1)]:
                    w = wx*wy*wz
                    rho[ix, iy, iz] += q * w
                    J[0, ix, iy, iz] += q * vx * w
                    J[1, ix, iy, iz] += q * vy * w
                    J[2, ix, iy, iz] += q * vz * w
    
    vol = dx**3
    rho /= vol
    J /= vol

def boris_pusher_3d(particles, E, B, dx, dt):
    nx = E.shape[1]
    ny = E.shape[2]
    nz = E.shape[3]
    
    for p in particles:
        x_cell = p.position[0]/dx
        y_cell = p.position[1]/dx
        z_cell = p.position[2]/dx
        
        i0 = int(np.floor(x_cell)) % nx
        j0 = int(np.floor(y_cell)) % ny
        k0 = int(np.floor(z_cell)) % nz
        
        fx = x_cell - i0
        fy = y_cell - j0
        fz = z_cell - k0
        
        i1 = (i0+1)%nx
        j1 = (j0+1)%ny
        k1 = (k0+1)%nz

        wx0 = 1 - fx
        wx1 = fx
        wy0 = 1 - fy
        wy1 = fy
        wz0 = 1 - fz
        wz1 = fz

        def tricubic(field):
            # Actually trilinear interpolation
            val = (field[i0,j0,k0]*wx0*wy0*wz0 +
                   field[i1,j0,k0]*wx1*wy0*wz0 +
                   field[i0,j1,k0]*wx0*wy1*wz0 +
                   field[i0,j0,k1]*wx0*wy0*wz1 +
                   field[i1,j1,k0]*wx1*wy1*wz0 +
                   field[i1,j0,k1]*wx1*wy0*wz1 +
                   field[i0,j1,k1]*wx0*wy1*wz1 +
                   field[i1,j1,k1]*wx1*wy1*wz1)
            return val
        
        E_part = np.array([tricubic(E[0]), tricubic(E[1]), tricubic(E[2])])
        B_part = np.array([tricubic(B[0]), tricubic(B[1]), tricubic(B[2])])

        # Boris algorithm
        qmdt = p.charge*dt/(2*p.mass)
        v_minus = p.velocity + qmdt*E_part
        
        t = qmdt * B_part
        t2 = np.dot(t,t)
        
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + (np.cross(v_prime, t)*2.0/(1+t2))
        
        # final velocity
        v_new = v_plus + qmdt*E_part
        
        p.velocity = v_new
        p.position += v_new * dt
        # periodic
        p.position[0] %= (nx*dx)
        p.position[1] %= (ny*dx)
        p.position[2] %= (nz*dx)

def maxwell_fdtd_3d(E, B, rho, J, dt, dx):
    nx, ny, nz = E.shape[1], E.shape[2], E.shape[3]
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    Jx, Jy, Jz = J

    def shift(arr, dx, axis):
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*dx)

    # After selecting Ex, Ey, Ez each is (nx, ny, nz)
    # Axes: 0 -> x, 1 -> y, 2 -> z

    # curl E = ∇ × E
    # For 3D arrays (nx, ny, nz):
    # d/dx -> axis=0, d/dy -> axis=1, d/dz -> axis=2
    dEz_dy = shift(Ez, dx, 1)  # derivative of Ez w.r.t y
    dEy_dz = shift(Ey, dx, 2)  # derivative of Ey w.r.t z
    dEx_dz = shift(Ex, dx, 2)  # derivative of Ex w.r.t z
    dEz_dx = shift(Ez, dx, 0)  # derivative of Ez w.r.t x
    dEy_dx = shift(Ey, dx, 0)  # derivative of Ey w.r.t x
    dEx_dy = shift(Ex, dx, 1)  # derivative of Ex w.r.t y

    curlE_x = dEz_dy - dEy_dz
    curlE_y = dEx_dz - dEz_dx
    curlE_z = dEy_dx - dEx_dy

    # curl B = ∇ × B
    dBz_dy = shift(Bz, dx, 1)
    dBy_dz = shift(By, dx, 2)
    dBx_dz = shift(Bx, dx, 2)
    dBz_dx = shift(Bz, dx, 0)
    dBy_dx = shift(By, dx, 0)
    dBx_dy = shift(Bx, dx, 1)

    curlB_x = dBz_dy - dBy_dz
    curlB_y = dBx_dz - dBz_dx
    curlB_z = dBy_dx - dBx_dy

    Ex_new = Ex + dt/epsilon_0*(curlB_x - Jx)
    Ey_new = Ey + dt/epsilon_0*(curlB_y - Jy)
    Ez_new = Ez + dt/epsilon_0*(curlB_z - Jz)

    Bx_new = Bx - dt/mu_0*(curlE_x)
    By_new = By - dt/mu_0*(curlE_y)
    Bz_new = Bz - dt/mu_0*(curlE_z)

    E_new = np.array([Ex_new, Ey_new, Ez_new])
    B_new = np.array([Bx_new, By_new, Bz_new])

    return E_new, B_new


# Simulation parameters
nx, ny, nz = 32, 32, 32
dx = 1e-2
dt = dx/(2*c)
steps = 1000

# Initialize fields
E = np.zeros((3, nx, ny, nz))
B = np.zeros((3, nx, ny, nz))
rho = np.zeros((nx, ny, nz))
J = np.zeros((3, nx, ny, nz))

# Initialize particles
particles = []
m_e = 9.11e-31
q_e = -1.602e-19

N = 100
v0 = 1e5
amplitude = 0

for i in range(N//2):
    x = np.random.rand()*nx*dx
    x += amplitude * dx * np.sin(2*np.pi*x/(nx*dx))
    y = np.random.rand()*ny*dx
    z = np.random.rand()*nz*dx
    vx = v0 + np.random.randn()*1e3
    vy = np.random.randn()*1e3
    vz = np.random.randn()*1e3
    particles.append(Particle([x % (nx*dx), y, z], [vx, vy, vz], q_e, m_e))

for i in range(N//2, N):
    x = np.random.rand()*nx*dx
    x += amplitude * dx * np.sin(2*np.pi*x/(nx*dx))
    y = np.random.rand()*ny*dx
    z = np.random.rand()*nz*dx
    vx = -v0 + np.random.randn()*1e3
    vy = np.random.randn()*1e3
    vz = np.random.randn()*1e3
    particles.append(Particle([x % (nx*dx), y, z], [vx, vy, vz], q_e, m_e))


output_filename = "positions.xyz"

# Run the simulation loop
with open(output_filename, "w") as f:
    for step in range(steps):
        # Deposit charge and current
        deposit_charge_current_3d(particles, rho, J, dx, dt)

        # Update fields using Maxwell-FDTD
        E, B = maxwell_fdtd_3d(E, B, rho, J, dt, dx)

        # Move particles with the Boris pusher
        boris_pusher_3d(particles, E, B, dx, dt)
        
        # Write positions in XYZ format
        # XYZ format:
        # First line: number of particles
        # Second line: comment (e.g. "Step X")
        # Then N lines of: AtomSymbol x y z
        f.write(f"{N}\n")
        f.write(f"Step {step}\n")
        for p in particles:
            x, y, z = p.position
            f.write(f"X {x:.6e} {y:.6e} {z:.6e}\n")
        
        # Print to console to show progress
        print(f"Step {step} completed.")
