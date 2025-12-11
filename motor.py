import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. parameter

# step length
a = 8.0                 # nm

# steps per second
S = 100.0               # 每秒大约 100 次尝试跳步

#R analytic parameters
T = 20
kb = 1.38 * 10**-23 
g_atp = 2**-22
# direction R
R = 21.3               # np.exp(g_atp/(kb*T))            

k_minus = S / (1.0 + R)
k_plus  = R * k_minus

print(f"k_plus  = {k_plus:.3f} s^-1")
print(f"k_minus = {k_minus:.3f} s^-1")

# theory: mean velocity and diffusion constant for one state
v_theory = a * (k_plus - k_minus)            # nm / s
D_theory = (a**2 / 2.0) * (k_plus + k_minus) # nm^2 / s

print(f"theory v = {v_theory:.2f} nm/s")
print(f"theory D = {D_theory:.2f} nm^2/s")

# ======================
# 2. simulation
# ======================

dt = 1e-3               # time step, satisfy (k+ + k-) * dt << 1
T_total = 10.0          # simulation time(s)
n_steps = int(T_total / dt)

n_traj = 50             # number trajectories

# checking probability 
p_jump = (k_plus + k_minus) * dt
print(f"(k_plus + k_minus)*dt = {p_jump:.4f}  (less than 1)")

# ======================
# 3. one-state biased random walk
# ======================

t = np.arange(n_steps) * dt
traj = np.zeros((n_traj, n_steps))

p_plus  = k_plus * dt
p_minus = k_minus * dt

for n in range(n_traj):
    x = 0.0  # initial position
    positions = np.zeros(n_steps)
    for i in range(1, n_steps):
        r = np.random.rand()
        if r < p_plus:
            x += a
        elif r < p_plus + p_minus:
            x -= a
        # otherwise stay
        positions[i] = x
    traj[n, :] = positions

mean_pos = traj.mean(axis=0)           # nm
var_pos = traj.var(axis=0)             # nm^2

# numeric velocity
v_num = (mean_pos[-1] - mean_pos[0]) / T_total
print(f"numeric v ≈ {v_num:.2f} nm/s")
print("R =", R)

plt.figure(figsize=(6,4))
for n in range(min(n_traj, 10)):
    plt.plot(t, traj[n], alpha=0.3)
plt.plot(t, mean_pos, 'k', linewidth=2, label='mean')
plt.xlabel('time (s)')
plt.ylabel('position (nm)')
plt.legend()
plt.title('One-state motor trajectories')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(t, var_pos, label='MSD (numeric)')
plt.plot(t, 2 * D_theory * t, 'r--', label='2 D t theory')
plt.xlabel('time (s)')
plt.ylabel('variance of position (nm^2)')
plt.legend()
plt.title('MSD vs 2Dt')
plt.tight_layout()
plt.show()


# ======================
# 6. animation：一条轨迹 + 移动小球
# ======================

from matplotlib.animation import FuncAnimation

# 选一条trajectory，这里就用第 0 条
x_traj = traj[0]
t_traj = t               # 为了可读性单独命名

fig, ax = plt.subplots(figsize=(7,4))

# 先设置好坐标范围
ax.set_xlim(0, T_total)
y_min = np.min(x_traj)
y_max = np.max(x_traj)
if y_min == y_max:       # 防止全程没怎么动导致 min == max
    y_min -= 10
    y_max += 10
ax.set_ylim(y_min - 10, y_max + 10)

ax.set_xlabel('time (s)')
ax.set_ylabel('position (nm)')
ax.set_title('Single motor trajectory (animation)')

# 曲线 + 小球（初始先空的）
line, = ax.plot([], [], lw=2, color='steelblue')
ball, = ax.plot([], [], 'ro', markersize=8)

def init():
    line.set_data([], [])
    ball.set_data([], [])
    return line, ball

def update(frame):
    # 画到当前 frame 为止的轨迹
    line.set_data(t_traj[:frame+1], x_traj[:frame+1])
    # 小球的位置也要是“序列”，哪怕只有一个点
    ball.set_data([t_traj[frame]], [x_traj[frame]])
    return line, ball

zoom_range = 30   # 显示 ±30 nm 的窗口，你可以调小或调大

window = 0.3  # seconds

def update_zoom(frame):
    line.set_data(t_traj[:frame+1], x_traj[:frame+1])
    ball.set_data([t_traj[frame]], [x_traj[frame]])

    # x-axis zoom into the recent 0.3s
    t_now = t_traj[frame]
    ax.set_xlim(t_now - window, t_now)

    return line, ball


ani = FuncAnimation(
    fig,
    update_zoom,
    frames=len(t_traj),   # 动画总帧数
    init_func=init,
    interval=20,          # 每帧间隔 ms，可以调快/调慢
    blit=True
)

ax.set_xlim(0, 11)
plt.tight_layout()
plt.show()


#----------------------------------------------------
# 选一条trajectory
x_traj = traj[0]   # 位置 (nm)
t_traj = t         # 时间 (s)

fig, ax = plt.subplots(figsize=(7, 2.5))

# ---- 轨道范围：只看 0~2000 nm ----
ax.set_xlim(0, 2000)     # 固定 zoom in 区间
ax.set_ylim(-1, 1)       # y 轴只用来画一条线

# 画出水平轨道
ax.axhline(0, color='k', linewidth=1)

# y 轴刻度没意义，可以隐藏
ax.set_yticks([])
ax.set_xlabel("Position (nm)")
ax.set_title("Motor on 1D track")

# 小球（初始为空）
ball, = ax.plot([], [], 'ro', markersize=10)

# ✅ 时间文字：只创建一次，然后每帧用 set_text 更新
time_text = ax.text(
    0.5, 1.15, "",           # (x, y) 用 axes 坐标系
    transform=ax.transAxes,
    ha='center', va='center',
    fontsize=12
)

def init():
    ball.set_data([], [])
    time_text.set_text("")   # 清空文字
    return ball, time_text

def update(frame):
    x_now = x_traj[frame]
    t_now = t_traj[frame]

    # 小球位置（注意用列表）
    ball.set_data([x_now], [0])

    # 只更新文字，不新建 text 对象
    time_text.set_text(f"t = {t_now:.2f} s")

    return ball, time_text

ani = FuncAnimation(
    fig,
    update,
    frames=len(t_traj),
    init_func=init,
    interval=20,   # 毫秒
    blit=True
)

plt.tight_layout()
plt.show()

