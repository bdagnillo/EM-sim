import numpy as np 
import sys
import math
from numba import njit, prange
import os
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
def deposit_charge_current_3d(positions, velocities, charges, rho, J, dx, dt):
    nx, ny, nz = rho.shape

    # Zero out rho and J arrays (thread-safe clearing)
    rho[:] = 0.0
    J[:] = 0.0

    # Local thread-safe accumulation for rho and J
    rho_local = np.zeros_like(rho)
    J_local = np.zeros_like(J)

    N = positions.shape[0]
    for p in prange(N):
        x = positions[p, 0]
        y = positions[p, 1]
        z = positions[p, 2]
        vx = velocities[p, 0]
        vy = velocities[p, 1]
        vz = velocities[p, 2]
        q = charges[p]

        x_cell = x / dx
        y_cell = y / dx
        z_cell = z / dx

        i0 = int(np.floor(x_cell))
        j0 = int(np.floor(y_cell))
        k0 = int(np.floor(z_cell))

        if i0 < 0 or i0 >= nx - 1 or j0 < 0 or j0 >= ny - 1 or k0 < 0 or k0 >= nz - 1:
            continue

        fx = x_cell - i0
        fy = y_cell - j0
        fz = z_cell - k0

        wx0, wx1 = 1 - fx, fx
        wy0, wy1 = 1 - fy, fy
        wz0, wz1 = 1 - fz, fz

        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1

        for (ix, wx) in [(i0, wx0), (i1, wx1)]:
            for (iy, wy) in [(j0, wy0), (j1, wy1)]:
                for (iz, wz) in [(k0, wz0), (k1, wz1)]:
                    w = wx * wy * wz
                    rho_local[ix, iy, iz] += q * w
                    J_local[0, ix, iy, iz] += q * vx * w
                    J_local[1, ix, iy, iz] += q * vy * w
                    J_local[2, ix, iy, iz] += q * vz * w

    # Thread-safe accumulation
    for ix in prange(nx):
        for iy in range(ny):
            for iz in range(nz):
                rho[ix, iy, iz] += rho_local[ix, iy, iz]
                for c in range(3):
                    J[c, ix, iy, iz] += J_local[c, ix, iy, iz]

    # Normalize by volume
    vol = dx**3
    rho /= vol
    J /= vol


@njit(parallel=True)
def maxwell_fdtd_3d(E, B, rho, J, dt, dx):
    nx, ny, nz = E.shape[1], E.shape[2], E.shape[3]
    Ex, Ey, Ez = E[0], E[1], E[2]
    Bx, By, Bz = B[0], B[1], B[2]
    Jx, Jy, Jz = J[0], J[1], J[2]

    # # Helper function for partial derivatives with periodic boundaries
    # def partial_diff(f, dx, axis):
    #     shifted_f_forward = np.roll(f, -1, axis=axis)
    #     shifted_f_backward = np.roll(f, 1, axis=axis)
    #     return (shifted_f_forward - shifted_f_backward) / (2 * dx)

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

    # Return updated fields
    return [Ex, Ey, Ez], [Bx, By, Bz]



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

# def initialize_fields():

# Simulation parameters
nx, ny, nz = 50, 50, 50
dx = 1e-8
# dt = dx/(2*c)*0.001
dt = dx/(2*c)
print(dt)
steps = 10000000

E = np.zeros((3, nx, ny, nz), dtype=np.float64)
B = np.zeros((3, nx, ny, nz), dtype=np.float64)
rho = np.zeros((nx, ny, nz), dtype=np.float64)
J = np.zeros((3, nx, ny, nz), dtype=np.float64)

#m_e = 9.11e-31
#q_e = -1.602e-19
m_e =  1.67e-27       # proton mass (~1836 times electron mass)
q_e = 1.602e-19       # proton charge
N_e = 1e9
Q_e = N_e * q_e
M_e = N_e * m_e

N_electrons = 1000
N_particles = N_electrons

positions = np.zeros((N_particles, 3), dtype=np.float64)
velocities = np.zeros((N_particles, 3), dtype=np.float64)
charges = np.zeros(N_particles, dtype=np.float64)
masses = np.zeros(N_particles, dtype=np.float64)

for i in range(N_electrons):
    x = np.random.rand()*nx*dx
    y = np.random.rand()*ny*dx
    z = np.random.rand()*nz*dx
    # x = nx*dx/2
    # y = ny*dx/2
    # z = nz*dx/2
    # print("Limits",nx*dx,ny*dx,nz*dx)
    # print("Positions",x,y,z)
    vx = np.random.randn()*1e5
    vy = np.random.randn()*1e5
    vz = np.random.randn()*1e5
    # vx = vy = 0

    positions[i,0] = x
    positions[i,1] = y
    positions[i,2] = z
    velocities[i,0] = vx
    velocities[i,1] = vy
    velocities[i,2] = vz
    charges[i] = Q_e *1e5
    masses[i] = M_e




def loop(step,path):
    global E, B
    deposit_charge_current_3d(positions, velocities, charges, rho, J, dx, dt)
    # print(rho.all())
    # print(E)
    # E, B = maxwell_fdtd_3d(E, B, rho, J, dt, dx)
    E,B = maxwell_fdtd_3d_v2(E,B,rho,J,dt,dx)
    E, B = np.array(E), np.array(B)
    boris_pusher_3d(positions, velocities, charges, masses, E, B, dx, dt)
        
    if step % 1000 == 0:
        f.write(f"{N_particles}\n")
        f.write(f"Step {step}\n")
        for i in range(N_particles):
            x, y, z = positions[i]
            f.write(f"X {x:.6e} {y:.6e} {z:.6e}\n")
        print(f"Step {step} completed.")
    
    return E[0], E[1], E[2], B[0], B[1], B[2]


if __name__ == "__main__":
    
    output_filename = "positions.xyz"
    
    if len(sys.argv) > 1 and sys.argv[1] == "animate":
        x, y, z = np.arange(len(E[0])), np.arange(len(E[1])), np.arange(len(E[2]))
        # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # Use 'ij' indexing for proper dimensional mapping
        X, Y, Z = np.meshgrid(x,y,z)
        
        fig = plt.figure(figsize=(12,6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        quiver_E= ax1.quiver(x, y, z, np.zeros_like(x), np.zeros_like(y), np.zeros_like(z), length=0.2, normalize=True)
        quiver_B = ax2.quiver(x, y, z, np.zeros_like(x), np.zeros_like(y), np.zeros_like(z), length=0.2, normalize=True)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Electric Field")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.set_title("Magnetic Field")
        
        f = open(output_filename,'w')
        def update(step): #function to update animation
            print(step)
            global quiver_E, quiver_B
            Ex, Ey, Ez, Bx, By, Bz = loop(step,f)
            quiver_E.remove()
            quiver_B.remove()
            
            quiver_E= ax1.quiver(x, y, z, Ex, Ey, Ez, length=0.2, normalize=True)
            quiver_B = ax2.quiver(x, y, z, Bx, By, Bz, length=0.2, normalize=True)
        
            return quiver_E, quiver_B,
        
        animation = FuncAnimation(fig, update, frames=steps,blit=False)
        writer = FFMpegWriter(fps=1)
        animation.save('animation.mp4', writer=writer)
        
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        x, y, z = np.arange(len(E[0])), np.arange(len(E[1])), np.arange(len(E[2]))
        # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # Use 'ij' indexing for proper dimensional mapping
        X, Y, Z = np.meshgrid(x,y,z)
        
        
        def plot_2d(Ex,Ey,Bx,By,z_slice):
            fig = plt.figure(figsize=(12,6))
        
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            Ex_2D = Ex[z_slice, :, :]
            Ey_2D = Ey[z_slice, :, :]
            
            Bx_2D = Bx[z_slice, :, :]
            By_2D = By[z_slice, :, :]
            
            # print(Ex_2D)
            stream_E= ax1.streamplot(x, y, Ex_2D[z], Ey_2D[z])
            stream_B = ax2.streamplot(x, y, Bx_2D[z], By_2D[z])
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_title("Electric Field")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_title("Magnetic Field")
            
            plt.show()
        
        f = open(output_filename,'w')
        for step in range(steps):
            Ex, Ey, Ez, Bx, By, Bz = loop(step,f)
            
            if step % 100 == 0:
                plot_2d(Ex,Ey,Bx,By,len(Bz)//2 + 5)
            
    if len(sys.argv) > 1 and sys.argv[1] == "savefields":
        
        def save_vector_fields(vector_field1, vector_field2, filename, resolution=0.1):
            # file = open(filename,'w')
            shape = vector_field1[0].shape
            slice_amount = int(resolution * (shape[1]))
            print("Slice amount =",slice_amount)
            # slice_dim1 = slice(1, 3)  # Slice along the first dimension (rows)
            # slice_dim2 = slice(0, 2)  # Slice along the second dimension (columns)
            # slice_dim3 = slice(1, 3)  # Slice along the third dimension (depth)

            a1 = vector_field1[::,::slice_amount,::slice_amount,::slice_amount]
            a2 = vector_field2[::,::slice_amount,::slice_amount,::slice_amount]
            a1,a2 = np.array(a1),np.array(a2)
            
            np.savez_compressed(filename, vector_field1=a1, vector_field2=a2)
            # file.close()
        
        
        def delete_all_files_in_folder(folder_path):
            
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"The folder {folder_path} does not exist.")

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    raise OSError(f"Error occurred while deleting {file_path}: {e}")
        
        with open(output_filename, 'w') as f:
            delete_all_files_in_folder("Fields")
            for step in range(steps):
                loop(step,f)
                
                if step % 500 == 0:
                    save_vector_fields(E,B,f"Fields/{step}")
        
    else:
        # default to no animation

        # Open the file once before the loop
        with open(output_filename, "w") as f:
            
            for step in range(steps):
                loop(step,f)

