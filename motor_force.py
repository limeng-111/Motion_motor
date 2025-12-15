import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================================
# 1. Global parameters (same as your style)
# =====================================

a_nm = 8.0                     # step length in nm
a_m  = a_nm * 1e-9             # step length in meters (for F*a work)
S = 100.0                      # total stepping rate (k+ + k-) [1/s]
R0 = 21.3                      # zero-force bias k+ / k- (your original R)

kB = 1.38e-23                  # J/K
T = 300.0                      # temperature [K]

dt = 1e-3                      # time step [s]
T_total = 10.0                 # total simulation time [s]
n_steps = int(T_total / dt)
n_traj = 50                    # number of trajectories per force

# five different forces in pN (opposing forward motion)
forces_pN = [0, 0.2, 0.4, 0.6, 0.8]
forces_N = [F * 1e-12 for F in forces_pN]


# =====================================
# 2. Helper: simulate motor trajectories for given k+ and k-
# =====================================

def simulate_motor(k_plus, k_minus, a_nm, dt, T_total, n_traj):
    n_steps = int(T_total / dt)
    t = np.arange(n_steps) * dt
    traj = np.zeros((n_traj, n_steps))

    p_plus = k_plus * dt
    p_minus = k_minus * dt

    if p_plus + p_minus > 1.0:
        print("WARNING: (k+ + k-) * dt > 1. Decrease dt.")

    for n in range(n_traj):
        x = 0.0
        positions = np.zeros(n_steps)
        for i in range(1, n_steps):
            r = np.random.rand()
            if r < p_plus:
                x += a_nm
            elif r < p_plus + p_minus:
                x -= a_nm
            # else: stay
            positions[i] = x
        traj[n, :] = positions

    mean_pos = traj.mean(axis=0)
    var_pos = traj.var(axis=0)
    return t, traj, mean_pos, var_pos


# =====================================
# 3. Loop over forces: compute rates, simulate, store trajectories
# =====================================

all_traj = []       # list of traj arrays, one per force
all_mean = []       # mean trajectories
all_vel = []        # numeric velocities

for F_pN, F in zip(forces_pN, forces_N):

    # Force-dependent bias:
    # R(F) = R0 * exp( - F * a / (kB T) )
    R_F = R0 * np.exp(-F * a_m / (kB * T))

    k_minus = S / (1.0 + R_F)
    k_plus  = R_F * k_minus

    print(f"\nForce = {F_pN} pN")
    print(f"  R(F)      = {R_F:.3f}")
    print(f"  k_plus    = {k_plus:.3f} s^-1")
    print(f"  k_minus   = {k_minus:.3f} s^-1")

    # theory drift and diffusion (for info)
    v_theory = a_nm * (k_plus - k_minus)                 # nm/s
    D_theory = (a_nm**2 / 2.0) * (k_plus + k_minus)      # nm^2/s
    print(f"  v_theory  = {v_theory:.2f} nm/s")
    print(f"  D_theory  = {D_theory:.2f} nm^2/s")

    t, traj, mean_pos, var_pos = simulate_motor(
        k_plus, k_minus, a_nm, dt, T_total, n_traj
    )

    v_num = (mean_pos[-1] - mean_pos[0]) / T_total
    print(f"  v_numeric = {v_num:.2f} nm/s")

    all_traj.append(traj)
    all_mean.append(mean_pos)
    all_vel.append(v_num)


# =====================================
# 4. Plot sample trajectories for each force
# =====================================

plt.figure(figsize=(10, 8))

n_forces = len(forces_pN)
for idx, (F_pN, traj, mean_pos) in enumerate(zip(forces_pN, all_traj, all_mean)):
    plt.subplot(2, 3, idx + 1)
    # plot a few sample trajectories
    for n in range(min(5, n_traj)):
        plt.plot(t, traj[n], alpha=0.4)
    # plot mean
    plt.plot(t, mean_pos, 'k', linewidth=2, label='mean')
    plt.title(f'{F_pN} pN')
    plt.xlabel('time (s)')
    plt.ylabel('position (nm)')
    plt.legend()

plt.suptitle('Motor trajectories under different forces')
plt.tight_layout()
plt.show()


# =====================================
# 5. Force–velocity curve
# =====================================

plt.figure(figsize=(6,4))
plt.plot(forces_pN, all_vel, 'o-', label='numeric v(F)')
plt.axhline(0, color='k', linewidth=1, alpha=0.5)
plt.xlabel('Force (pN)')
plt.ylabel('Velocity (nm/s)')
plt.title('Force–velocity relationship (one-state motor)')
plt.legend()
plt.tight_layout()
plt.show()


# =====================================
# 6. OPTIONAL: 1D track animation for ONE chosen force
# =====================================

# choose which force index to animate (0..4)
idx_anim = 2   # e.g. index 2 -> force = 4 pN

x_traj = all_traj[idx_anim][0]   # take the first trajectory at that force
F_anim = forces_pN[idx_anim]

fig, ax = plt.subplots(figsize=(7, 2.5))

ax.set_xlim(0, 2000)  # zoom in spatially
ax.set_ylim(-1, 1)
ax.axhline(0, color='k', linewidth=1)
ax.set_yticks([])
ax.set_xlabel("Position (nm)")
ax.set_title(f"Motor on 1D track (F = {F_anim} pN)")

ball, = ax.plot([], [], 'ro', markersize=10)
time_text = ax.text(
    0.5, 1.15, "",
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=12
)

def init_anim():
    ball.set_data([], [])
    time_text.set_text("")
    return ball, time_text

def update_anim(frame):
    x_now = x_traj[frame]
    t_now = t[frame]
    ball.set_data([x_now], [0])
    time_text.set_text(f"t = {t_now:.2f} s")
    return ball, time_text

duration = 1.5  # seconds
frames = np.where(t <= duration)[0]
ani = FuncAnimation(
    fig,
    update_anim,
    frames=frames,
    init_func=init_anim,
    interval=20,
    blit=True
)

plt.tight_layout()
ani.save("motor_force.gif", writer="pillow", fps=30)

plt.show()

# If you want a GIF:
# ani.save("motor_force_%dpN.gif" % F_anim, writer="pillow", fps=30)
