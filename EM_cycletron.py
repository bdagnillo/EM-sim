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
    # For a single particle test, this won't significantly change anything
    # since there's no charge imbalance to drive fields, but we keep it for consistency.
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
        
        v_new = v_plus + qmdt*E_part
        
        p.velocity = v_new
        p.position += v_new * dt
        # If you want to remove periodic BC, comment out these lines:
        # p.position[0] %= (nx*dx)
        # p.position[1] %= (ny*dx)
        # p.position[2] %= (nz*dx)

def maxwell_fdtd_3d(E, B, rho, J, dt, dx):
    nx, ny, nz = E.shape[1], E.shape[2], E.shape[3]
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    Jx, Jy, Jz = J

    def shift(arr, dx, axis):
        return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2*dx)

    # For a uniform B field and no net currents, fields won't change much.
    # Still, we do the calculation for completeness.

    dEz_dy = shift(Ez, dx, 1)
    dEy_dz = shift(Ey, dx, 2)
    dEx_dz = shift(Ex, dx, 2)
    dEz_dx = shift(Ez, dx, 0)
    dEy_dx = shift(Ey, dx, 0)
    dEx_dy = shift(Ex, dx, 1)

    curlE_x = dEz_dy - dEy_dz
    curlE_y = dEx_dz - dEz_dx
    curlE_z = dEy_dx - dEx_dy

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

# Cyclotron test setup
nx, ny, nz = 32, 32, 32
dx = 1e-2
dt = dx/(2*c)*0.01
steps = 1000000
# Initialize fields: zero E, uniform B along z
E = np.zeros((3, nx, ny, nz))
B = np.zeros((3, nx, ny, nz))
B0 = 1e-3  # Tesla
B[2,:,:,:] = B0  # B in z-direction

rho = np.zeros((nx, ny, nz))
J = np.zeros((3, nx, ny, nz))

# Single electron
m_e = 9.11e-31
q_e = -1.602e-19
N = 1

# Place particle in center
x0 = (nx*dx)/2
y0 = (ny*dx)/2
z0 = (nz*dx)/2

# Initial velocity perpendicular to B (along x)
v_perp = 1e6 # m/s
particle = Particle([x0, y0, z0], [v_perp, 0.0, 0.0], q_e, m_e)
particles = [particle]

output_filename = "cyclotron_positions.xyz"

with open(output_filename, "w") as f:
    for step in range(steps):
        deposit_charge_current_3d(particles, rho, J, dx, dt)
        E, B = maxwell_fdtd_3d(E, B, rho, J, dt, dx)
        boris_pusher_3d(particles, E, B, dx, dt)
        
        f.write("1\n")
        f.write(f"Step {step}\n")
        x, y, z = particles[0].position
        f.write(f"X {x:.6e} {y:.6e} {z:.6e}\n")
        
        if step % 100 == 0:
            print(f"Step {step}: x={x:.3e}, y={y:.3e}, z={z:.3e}")
