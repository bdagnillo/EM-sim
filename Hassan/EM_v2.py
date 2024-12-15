import numpy as np 
import sys
import math
from numba import njit, prange

# Physical constants
epsilon_0 = 8.854187817e-12
mu_0 = 1.2566370614e-6
c = 1/np.sqrt(mu_0*epsilon_0)

@njit
def trilinear_interpolation(field, i0, j0, k0, i1, j1, k1, wx0, wy0, wz0, wx1, wy1, wz1):
    return (field[i0,j0,k0]*wx0*wy0*wz0 +
            field[i1,j0,k0]*wx1*wy0*wz0 +
            field[i0,j1,k0]*wx0*wy1*wz0 +
            field[i0,j0,k1]*wx0*wy0*wz1 +
            field[i1,j1,k0]*wx1*wy1*wz0 +
            field[i1,j0,k1]*wx1*wy0*wz1 +
            field[i0,j1,k1]*wx0*wy1*wz1 +
            field[i1,j1,k1]*wx1*wy1*wz1)

@njit(parallel=True)
def deposit_charge_current_3d(positions, velocities, charges, rho, J, dx, dt):
    nx, ny, nz = rho.shape
    for ix in prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                rho[ix, iy, iz] = 0.0
    for c in range(3):
        for ix in prange(nx):
            for iy in range(ny):
                for iz in range(nz):
                    J[c, ix, iy, iz] = 0.0

    N = positions.shape[0]
    for p in prange(N):
        x = positions[p,0]
        y = positions[p,1]
        z = positions[p,2]
        vx = velocities[p,0]
        vy = velocities[p,1]
        vz = velocities[p,2]
        q = charges[p]

        x_cell = x/dx
        y_cell = y/dx
        z_cell = z/dx

        i0 = int(math.floor(x_cell))
        j0 = int(math.floor(y_cell))
        k0 = int(math.floor(z_cell))

        # No PBC: just clamp indices if needed, or assume particles never leave domain.
        # For safety, clamp indices:
        if i0 < 0 or i0 >= nx-1: 
            continue
        if j0 < 0 or j0 >= ny-1:
            continue
        if k0 < 0 or k0 >= nz-1:
            continue

        fx = x_cell - i0
        fy = y_cell - j0
        fz = z_cell - k0

        wx0 = 1 - fx
        wx1 = fx
        wy0 = 1 - fy
        wy1 = fy
        wz0 = 1 - fz
        wz1 = fz

        i1 = i0+1
        j1 = j0+1
        k1 = k0+1

        for (ix, wx) in [(i0, wx0), (i1, wx1)]:
            for (iy, wy) in [(j0, wy0), (j1, wy1)]:
                for (iz, wz) in [(k0, wz0), (k1, wz1)]:
                    w = wx*wy*wz
                    rho[ix, iy, iz] += q * w
                    J[0, ix, iy, iz] += q * vx * w
                    J[1, ix, iy, iz] += q * vy * w
                    J[2, ix, iy, iz] += q * vz * w

    vol = dx**3
    for ix in prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                rho[ix, iy, iz] /= vol
    for c in range(3):
        for ix in prange(nx):
            for iy in range(ny):
                for iz in range(nz):
                    J[c, ix, iy, iz] /= vol

@njit(parallel=True)
def boris_pusher_3d(positions, velocities, charges, masses, E, B, dx, dt):
    nx = E.shape[1]
    ny = E.shape[2]
    nz = E.shape[3]

    N = positions.shape[0]
    for p in prange(N):
        x = positions[p,0]
        y = positions[p,1]
        z = positions[p,2]

        x_cell = x/dx
        y_cell = y/dx
        z_cell = z/dx

        i0 = int(math.floor(x_cell))
        j0 = int(math.floor(y_cell))
        k0 = int(math.floor(z_cell))

        # If particle goes out of domain (no PBC), skip update or handle accordingly.
        if i0 < 0 or i0 >= nx-1 or j0 < 0 or j0 >= ny-1 or k0 < 0 or k0 >= nz-1:
            # Just skip if outside domain
            continue

        fx = x_cell - i0
        fy = y_cell - j0
        fz = z_cell - k0

        i1 = i0+1
        j1 = j0+1
        k1 = k0+1

        wx0 = 1 - fx
        wx1 = fx
        wy0 = 1 - fy
        wy1 = fy
        wz0 = 1 - fz
        wz1 = fz

        Ex_val = trilinear_interpolation(E[0], i0,j0,k0,i1,j1,k1, wx0,wy0,wz0,wx1,wy1,wz1)
        Ey_val = trilinear_interpolation(E[1], i0,j0,k0,i1,j1,k1, wx0,wy0,wz0,wx1,wy1,wz1)
        Ez_val = trilinear_interpolation(E[2], i0,j0,k0,i1,j1,k1, wx0,wy0,wz0,wx1,wy1,wz1)

        Bx_val = trilinear_interpolation(B[0], i0,j0,k0,i1,j1,k1, wx0,wy0,wz0,wx1,wy1,wz1)
        By_val = trilinear_interpolation(B[1], i0,j0,k0,i1,j1,k1, wx0,wy0,wz0,wx1,wy1,wz1)
        Bz_val = trilinear_interpolation(B[2], i0,j0,k0,i1,j1,k1, wx0,wy0,wz0,wx1,wy1,wz1)

        q = charges[p]
        m = masses[p]

        vx = velocities[p,0]
        vy = velocities[p,1]
        vz = velocities[p,2]

        qmdt = q*dt/(2*m)
        v_minus_x = vx + qmdt*Ex_val
        v_minus_y = vy + qmdt*Ey_val
        v_minus_z = vz + qmdt*Ez_val

        t_x = qmdt * Bx_val
        t_y = qmdt * By_val
        t_z = qmdt * Bz_val
        t2 = t_x*t_x + t_y*t_y + t_z*t_z

        cross_x = v_minus_y*t_z - v_minus_z*t_y
        cross_y = v_minus_z*t_x - v_minus_x*t_z
        cross_z = v_minus_x*t_y - v_minus_y*t_x

        v_prime_x = v_minus_x + cross_x
        v_prime_y = v_minus_y + cross_y
        v_prime_z = v_minus_z + cross_z

        cross2_x = v_prime_y*t_z - v_prime_z*t_y
        cross2_y = v_prime_z*t_x - v_prime_x*t_z
        cross2_z = v_prime_x*t_y - v_prime_y*t_x

        factor = 2.0/(1+t2)
        v_plus_x = v_minus_x + factor*cross2_x
        v_plus_y = v_minus_y + factor*cross2_y
        v_plus_z = v_minus_z + factor*cross2_z

        v_new_x = v_plus_x + qmdt*Ex_val
        v_new_y = v_plus_y + qmdt*Ey_val
        v_new_z = v_plus_z + qmdt*Ez_val

        velocities[p,0] = v_new_x
        velocities[p,1] = v_new_y
        velocities[p,2] = v_new_z

        # No PBC:
        x_new = x + v_new_x*dt
        y_new = y + v_new_y*dt
        z_new = z + v_new_z*dt

        positions[p,0] = x_new
        positions[p,1] = y_new
        positions[p,2] = z_new

@njit(parallel=True)
def partial_diff(arr, dx, axis):
    nx, ny, nz = arr.shape
    out = np.zeros((nx, ny, nz), dtype=np.float64)
    # No PBC: we assume fields vanish outside or just no indexing outside domain
    # We'll do a one-sided difference at boundaries.
    if axis == 0:
        for x in range(nx):
            xp = x+1 if x+1<nx else x
            xm = x-1 if x-1>=0 else x
            for y in range(ny):
                for z in range(nz):
                    out[x,y,z] = (arr[xp,y,z]-arr[xm,y,z])/(2*dx) if xp!=xm else 0.0
    elif axis == 1:
        for x in range(nx):
            for y in range(ny):
                yp = y+1 if y+1<ny else y
                ym = y-1 if y-1>=0 else y
                for z in range(nz):
                    out[x,y,z] = (arr[x,yp,z]-arr[x,ym,z])/(2*dx) if yp!=ym else 0.0
    else:
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    zp = z+1 if z+1<nz else z
                    zm = z-1 if z-1>=0 else z
                    out[x,y,z] = (arr[x,y,zp]-arr[x,y,zm])/(2*dx) if zp!=zm else 0.0
    return out

@njit(parallel=True)
def maxwell_fdtd_3d(E, B, rho, J, dt, dx):
    # Existing implementation remains the same
    # Numba will now parallelize the loops over grid points
    nx, ny, nz = E.shape[1], E.shape[2], E.shape[3]
    Ex, Ey, Ez = E[0], E[1], E[2]
    Bx, By, Bz = B[0], B[1], B[2]
    Jx, Jy, Jz = J[0], J[1], J[2]

    # Compute curls
    curlE_x = partial_diff(Ez, dx, 1) - partial_diff(Ey, dx, 2)
    curlE_y = partial_diff(Ex, dx, 2) - partial_diff(Ez, dx, 0)
    curlE_z = partial_diff(Ey, dx, 0) - partial_diff(Ex, dx, 1)

    curlB_x = partial_diff(Bz, dx, 1) - partial_diff(By, dx, 2)
    curlB_y = partial_diff(Bx, dx, 2) - partial_diff(Bz, dx, 0)
    curlB_z = partial_diff(By, dx, 0) - partial_diff(Bx, dx, 1)

    # Update E fields
    for ix in prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                Ex[ix, iy, iz] += dt / epsilon_0 * (curlB_x[ix, iy, iz] - Jx[ix, iy, iz])
                Ey[ix, iy, iz] += dt / epsilon_0 * (curlB_y[ix, iy, iz] - Jy[ix, iy, iz])
                Ez[ix, iy, iz] += dt / epsilon_0 * (curlB_z[ix, iy, iz] - Jz[ix, iy, iz])

    # Update B fields
    for ix in prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                Bx[ix, iy, iz] -= dt / mu_0 * curlE_x[ix, iy, iz]
                By[ix, iy, iz] -= dt / mu_0 * curlE_y[ix, iy, iz]
                Bz[ix, iy, iz] -= dt / mu_0 * curlE_z[ix, iy, iz]

    return E, B

@njit(parallel=True)
def maxwell_fdtd_3d_v2(E, B, rho, J, dt, dx):
    c = 3*1e8  # Assume natural units where c = 1

    Nx, Ny, Nz = E.shape[1], E.shape[2], E.shape[3]

    # Update magnetic field (Faraday's Law)
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                B[0, i, j, k] -= dt * (
                    (E[2, i, (j + 1) % Ny, k] - E[2, i, j, k]) / dx -
                    (E[1, i, j, (k + 1) % Nz] - E[1, i, j, k]) / dx
                )
                B[1, i, j, k] -= dt * (
                    (E[0, i, j, (k + 1) % Nz] - E[0, i, j, k]) / dx -
                    (E[2, (i + 1) % Nx, j, k] - E[2, i, j, k]) / dx
                )
                B[2, i, j, k] -= dt * (
                    (E[1, (i + 1) % Nx, j, k] - E[1, i, j, k]) / dx -
                    (E[0, i, (j + 1) % Ny, k] - E[0, i, j, k]) / dx
                )

    # Update electric field (Ampere's Law)
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                E[0, i, j, k] += dt * (
                    c**2 * ((B[2, i, (j + 1) % Ny, k] - B[2, i, j, k]) / dx -
                            (B[1, i, j, (k + 1) % Nz] - B[1, i, j, k]) / dx) -
                    J[0, i, j, k] + rho[i, j, k] / dt
                )
                E[1, i, j, k] += dt * (
                    c**2 * ((B[0, i, j, (k + 1) % Nz] - B[0, i, j, k]) / dx -
                            (B[2, (i + 1) % Nx, j, k] - B[2, i, j, k]) / dx) -
                    J[1, i, j, k] + rho[i, j, k] / dt
                )
                E[2, i, j, k] += dt * (
                    c**2 * ((B[1, (i + 1) % Nx, j, k] - B[1, i, j, k]) / dx -
                            (B[0, i, (j + 1) % Ny, k] - B[0, i, j, k]) / dx) -
                    J[2, i, j, k] + rho[i, j, k] / dt
                )

    return E, B


# Simulation parameters
nx, ny, nz = 32, 32, 32
dx = 1e-15
dt = dx/(2*c)
steps = 10000000

E = np.zeros((3, nx, ny, nz), dtype=np.float64)
B = np.zeros((3, nx, ny, nz), dtype=np.float64)
rho = np.zeros((nx, ny, nz), dtype=np.float64)
J = np.zeros((3, nx, ny, nz), dtype=np.float64)

m_e = 9.11e-31
q_e = -1.602e-19
m_p =  1.67e-27       # proton mass (~1836 times electron mass)
q_p = 1.602e-19         # proton charge
N_e = 1e9
Q_e = N_e * q_e
M_e = N_e * m_e
a_0 = 5.29e-11 # bohr radius

N_electrons = 1
N_protons = 1
N_particles = 2

positions = np.zeros((N_particles, 3), dtype=np.float64)
velocities = np.zeros((N_particles, 3), dtype=np.float64)
charges = np.zeros(N_particles, dtype=np.float64)
masses = np.zeros(N_particles, dtype=np.float64)

# for i in range(N_electrons):
#     x = np.random.rand()*nx*dx
#     y = np.random.rand()*ny*dx
#     z = np.random.rand()*nz*dx
#     vx = np.random.randn()*1e5
#     vy = np.random.randn()*1e5
#     vz = np.random.randn()*1e5
#     positions[i,0] = x
#     positions[i,1] = y
#     positions[i,2] = z
#     velocities[i,0] = vx
#     velocities[i,1] = vy
#     velocities[i,2] = vz
#     charges[i] = Q_e
#     masses[i] = M_e

# proton
positions[0] = [nx*dx/2, ny*dx/2, nz*dx/2]
velocities[0] = [0,0,0]
charges[0] = q_p
masses[0] = m_p
# electron
orbit_velocity = np.sqrt(q_e**2/(4 * np.pi * epsilon_0 * a_0 * m_e))
positions[1] = [nx*dx/2 + a_0, ny*dx/2, nz*dx/2]
velocities[1] = [0,orbit_velocity,0]
charges[1] = q_p
masses[1] = m_p


output_filename = "positions.xyz"

# Open the file once before the loop
with open(output_filename, "w") as f:
    for step in range(steps):
        deposit_charge_current_3d(positions, velocities, charges, rho, J, dx, dt)
        # E, B = maxwell_fdtd_3d(E, B, rho, J, dt, dx)
        E, B = maxwell_fdtd_3d_v2(E, B, rho, J, dt, dx)
        boris_pusher_3d(positions, velocities, charges, masses, E, B, dx, dt)
    
        if step % 500 == 0:
            f.write(f"{N_particles}\n")
            f.write(f"Step {step}\n")
            for i in range(N_particles):
                x, y, z = positions[i]
                f.write(f"X {x:.6e} {y:.6e} {z:.6e}\n")
            print(f"Step {step} completed.")
