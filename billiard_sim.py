import os
import io
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from PIL import Image
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def table_surface(x, y, W, H, k, epsilon):
    """
    Implicit surface function defining the fully asymmetric/chaotic billiard boundary
    with an internal asymmetric hole. Returns < 0 for points inside playable area.
    """
    # Outer boundary
    theta_out = np.arctan2(y, x)
    r_ell_out = 1.0 / np.sqrt((np.cos(theta_out) / (W / 2.0))**2 + (np.sin(theta_out) / (H / 2.0))**2)
    r_out = r_ell_out * (1.0 + epsilon * (np.sin(theta_out) + 0.8 * np.cos(k * theta_out - 1.2) - 0.5 * np.sin((k + 1) * theta_out + 2.0)))
    f_outer = np.hypot(x, y) - r_out
    
    # Inner hole boundary
    offset_x = W * 0.05
    offset_y = H * 0.05
    dx = x - offset_x
    dy = y - offset_y
    theta_in = np.arctan2(dy, dx)
    
    # Hole is scaled down to 35% size
    r_ell_in = 1.0 / np.sqrt((np.cos(theta_in) / (W * 0.35 / 2.0))**2 + (np.sin(theta_in) / (H * 0.35 / 2.0))**2)
    k_in = 1  # 1 ripple for the hole
    r_in = r_ell_in * (1.0 + epsilon * (np.sin(theta_in) + 0.8 * np.cos(k_in * theta_in - 1.2) - 0.5 * np.sin((k_in + 1) * theta_in + 2.0)))
    f_inner = np.hypot(dx, dy) - r_in
    
    # Playable area is intersection of (inside outer) AND (outside inner)
    # Boolean intersection of implicit functions: max(A, B) < 0
    return max(f_outer, -f_inner)

def get_normal(x, y, W, H, k, epsilon):
    """
    Computes the outward normal vector at a surface point using the numerical gradient.
    """
    h = 1e-6
    dx = (table_surface(x + h, y, W, H, k, epsilon) - table_surface(x - h, y, W, H, k, epsilon)) / (2 * h)
    dy = (table_surface(x, y + h, W, H, k, epsilon) - table_surface(x, y - h, W, H, k, epsilon)) / (2 * h)
    n = np.array([dx, dy])
    return n / np.linalg.norm(n)

def get_next_collision(P, V, W, H, k, epsilon):
    """
    Finds the next collision time and normal using ray-marching and bisection.
    """
    # Use an even finer step size due to the internal hole and tight margins
    t_step = 0.01 / np.linalg.norm(V)
    t = 1e-5 # Offset slightly to prevent immediate re-collision with the current wall
    
    # Ray-march forward until a boundary crossing is detected
    for _ in range(30000): 
        t_next = t + t_step
        p_next = P + V * t_next
        
        if table_surface(p_next[0], p_next[1], W, H, k, epsilon) > 0:
            # Boundary crossed. Isolate the exact root using bisection.
            t_low = t
            t_high = t_next
            for _ in range(45): 
                t_mid = (t_low + t_high) / 2.0
                p_mid = P + V * t_mid
                if table_surface(p_mid[0], p_mid[1], W, H, k, epsilon) > 0:
                    t_high = t_mid
                else:
                    t_low = t_mid
                    
            t_hit = (t_low + t_high) / 2.0
            p_hit = P + V * t_hit
            n = get_normal(p_hit[0], p_hit[1], W, H, k, epsilon)
            
            # The normal must point inwards for reflection
            return t_hit, -n 
        t = t_next
        
    return float('inf'), np.array([1.0, 0.0])

def generate_events(W, H, ripples, epsilon, speed, max_time):
    """
    Pre-calculates all collision events up to max_time.
    """
    # Start position shifted to ensure it spawns safely outside the new inner hole
    P = np.array([-W / 3.0, 0.0])
    theta = np.radians(37) 
    V = np.array([np.cos(theta), np.sin(theta)]) * speed
    
    times = [0.0]
    positions = [P.copy()]
    velocities = [V.copy()]
    
    current_time = 0.0
    while current_time < max_time:
        dt, normal = get_next_collision(P, V, W, H, ripples, epsilon)
        if dt == float('inf'):
            break 
            
        P = P + V * dt
        V = V - 2 * np.dot(V, normal) * normal 
        current_time += dt
        
        times.append(current_time)
        positions.append(P.copy())
        velocities.append(V.copy())
        
    return np.array(times), np.array(positions), np.array(velocities)

def evaluate_trajectory(times, positions, velocities, t_eval):
    """
    Interpolates the exact position at any given time array t_eval.
    """
    idx = np.searchsorted(times, t_eval) - 1
    idx = np.clip(idx, 0, len(times) - 1)
    dt = t_eval - times[idx]
    
    pos_x = positions[idx, 0] + velocities[idx, 0] * dt
    pos_y = positions[idx, 1] + velocities[idx, 1] * dt
    return np.column_stack((pos_x, pos_y))

def draw_boundary(ax, W, H, k, epsilon):
    """
    Plots the parametric boundary matching the implicit surface function
    and fills the playable area with a translucent gray color.
    """
    theta = np.linspace(0, 2 * np.pi, 1000)
    
    # 1. Outer boundary
    r_ell_out = 1.0 / np.sqrt((np.cos(theta) / (W / 2.0))**2 + (np.sin(theta) / (H / 2.0))**2)
    r_out = r_ell_out * (1.0 + epsilon * (np.sin(theta) + 0.8 * np.cos(k * theta - 1.2) - 0.5 * np.sin((k + 1) * theta + 2.0)))
    x_out = r_out * np.cos(theta)
    y_out = r_out * np.sin(theta)
    
    # 2. Inner hole boundary
    offset_x = W * 0.05
    offset_y = H * 0.05
    k_in = 1
    r_ell_in = 1.0 / np.sqrt((np.cos(theta) / (W * 0.35 / 2.0))**2 + (np.sin(theta) / (H * 0.35 / 2.0))**2)
    r_in = r_ell_in * (1.0 + epsilon * (np.sin(theta) + 0.8 * np.cos(k_in * theta - 1.2) - 0.5 * np.sin((k_in + 1) * theta + 2.0)))
    x_in = r_in * np.cos(theta) + offset_x
    y_in = r_in * np.sin(theta) + offset_y
    
    # Plot both boundary lines
    ax.plot(x_out, y_out, color='#d0d7de', linewidth=1.2)
    ax.plot(x_in, y_in, color='#d0d7de', linewidth=1.2)
    
    # Fill the playable area (polygon with a hole) using PathPatch
    verts_out = np.column_stack((x_out, y_out))
    # Reverse the inner vertices to create the cutout correctly
    verts_in = np.column_stack((x_in, y_in))[::-1]
    
    verts = np.vstack((verts_out, verts_in))
    codes = np.full(len(verts), mpath.Path.LINETO)
    codes[0] = mpath.Path.MOVETO
    codes[len(verts_out)] = mpath.Path.MOVETO
    
    path = mpath.Path(verts, codes)
    # Lighter gray base with slightly lower opacity for maximum tracer contrast
    patch = mpatches.PathPatch(path, facecolor='#4A4F55', edgecolor='none')
    ax.add_patch(patch)

def main():
    parser = argparse.ArgumentParser(description="Simulate 2D dynamical billiard with a chaotic organic boundary and internal hole.")
    parser.add_argument("--frames", type=int, default=1500, help="Total number of frames.")
    parser.add_argument("--fps", type=int, default=30, help="Animation frames per second.")
    parser.add_argument("--speed", type=float, default=6.0, help="Particle speed in units per second.")
    parser.add_argument("--width", type=float, default=14.0, help="Table base width.")
    parser.add_argument("--height", type=float, default=7.0, help="Table base height.")
    parser.add_argument("--ripples", type=int, default=5, help="Base frequency for organic ripples.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Amplitude of the boundary ripples.")
    parser.add_argument("--trail", type=float, default=1.2, help="Tail length in seconds.")
    parser.add_argument("--output", type=str, default="images/billiard_organic_hole_v1.gif", help="Output path.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dt = 1.0 / args.fps
    max_time = args.frames * dt
    evt_times, evt_pos, evt_vel = generate_events(args.width, args.height, args.ripples, args.epsilon, args.speed, max_time + args.trail)

    fig, ax = plt.subplots(figsize=(args.width / 2.0, args.height / 2.0), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # Strip Matplotlib's default figure padding
    
    # Tight margins based on chaotic shape's max amplitude (~2.3x epsilon) + 5% safe padding
    margin_factor = args.epsilon * 2.3 + 0.05
    margin_x = (args.width / 2.0) * margin_factor
    margin_y = (args.height / 2.0) * margin_factor
    ax.set_xlim(-args.width / 2.0 - margin_x, args.width / 2.0 + margin_x)
    ax.set_ylim(-args.height / 2.0 - margin_y, args.height / 2.0 + margin_y)
    
    draw_boundary(ax, args.width, args.height, args.ripples, args.epsilon)

    trail_resolution = int(args.trail * args.fps * 2) 
    # Widened the line width from 1.5 to 2.5
    line_collection = LineCollection([], linewidths=2.5, capstyle='round', joinstyle='round')
    ax.add_collection(line_collection)

    # Changed to an ultra-bright fluorescent cyan-blue to make it pop aggressively
    base_color = to_rgba('#00e5ff')
    rgba_colors = np.zeros((trail_resolution - 1, 4))
    rgba_colors[:, :3] = base_color[:3]
    rgba_colors[:, 3] = np.linspace(0.0, 1.0, trail_resolution - 1)**2 

    def update(frame):
        t_current = frame * dt
        t_start = max(0, t_current - args.trail)
        
        eval_times = np.linspace(t_start, t_current, trail_resolution)
        points = evaluate_trajectory(evt_times, evt_pos, evt_vel, eval_times)
        
        segments = np.zeros((trail_resolution - 1, 2, 2))
        segments[:, 0, :] = points[:-1, :]
        segments[:, 1, :] = points[1:, :]
        
        line_collection.set_segments(segments)
        line_collection.set_color(rgba_colors)

    print("Rendering frames to memory buffer. This will take a moment...")
    frames = []
    for i in range(args.frames):
        update(i)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', transparent=True, facecolor='none', edgecolor='none')
        buf.seek(0)
        frames.append(Image.open(buf))
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{args.frames} frames")

    print(f"Saving transparent animation to {args.output}...")
    
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        disposal=2, 
        transparency=0 
    )
    print("Export complete.")

if __name__ == "__main__":
    main()