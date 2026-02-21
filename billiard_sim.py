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

def table_surface(x, y, W, H, k, epsilon):
    """
    Implicit surface function defining the billiard boundary.
    Returns < 0 for points inside, > 0 for points outside.
    """
    theta = np.arctan2(y, x)
    
    # Base ellipse radius at angle theta
    r_ellipse = 1.0 / np.sqrt((np.cos(theta) / (W / 2.0))**2 + (np.sin(theta) / (H / 2.0))**2)
    
    # Perturb the radius with a sinusoid to create "whacky" ripples
    r_boundary = r_ellipse * (1.0 + epsilon * np.cos(k * theta))
    
    # Current distance from origin
    r_current = np.hypot(x, y)
    
    return r_current - r_boundary

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
    t_step = 0.05 / np.linalg.norm(V)
    t = 1e-5 # Offset slightly to prevent immediate re-collision with the current wall
    
    # Ray-march forward until a boundary crossing is detected
    for _ in range(10000): 
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
    P = np.array([0.0, 0.0])
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
    Plots the parametric boundary matching the implicit surface function.
    """
    theta = np.linspace(0, 2 * np.pi, 500)
    r_ellipse = 1.0 / np.sqrt((np.cos(theta) / (W / 2.0))**2 + (np.sin(theta) / (H / 2.0))**2)
    r_boundary = r_ellipse * (1.0 + epsilon * np.cos(k * theta))
    
    x = r_boundary * np.cos(theta)
    y = r_boundary * np.sin(theta)
    
    ax.plot(x, y, color='#d0d7de', linewidth=1.6)

def main():
    parser = argparse.ArgumentParser(description="Simulate 2D dynamical billiard with curved boundaries.")
    parser.add_argument("--frames", type=int, default=1500, help="Total number of frames.")
    parser.add_argument("--fps", type=int, default=30, help="Animation frames per second.")
    parser.add_argument("--speed", type=float, default=6.0, help="Particle speed in units per second.")
    parser.add_argument("--width", type=float, default=22.0, help="Table base width.")
    parser.add_argument("--height", type=float, default=14.0, help="Table base height.")
    parser.add_argument("--ripples", type=int, default=5, help="Number of sinusoidal ripples along the perimeter.")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Amplitude of the boundary ripples (0 to 1).")
    parser.add_argument("--trail", type=float, default=1.2, help="Tail length in seconds.")
    parser.add_argument("--output", type=str, default="images/dynamical_billiard_v3.gif", help="Output path.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dt = 1.0 / args.fps
    max_time = args.frames * dt
    evt_times, evt_pos, evt_vel = generate_events(args.width, args.height, args.ripples, args.epsilon, args.speed, max_time + args.trail)

    fig, ax = plt.subplots(figsize=(args.width / 2.0, args.height / 2.0), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.axis('off')
    
    # Provide a buffer margin to accommodate the ripples
    margin = args.width * args.epsilon
    ax.set_xlim(-args.width / 2.0 - margin, args.width / 2.0 + margin)
    ax.set_ylim(-args.height / 2.0 - margin, args.height / 2.0 + margin)
    
    draw_boundary(ax, args.width, args.height, args.ripples, args.epsilon)

    trail_resolution = int(args.trail * args.fps * 2) 
    line_collection = LineCollection([], linewidths=1.2, capstyle='round', joinstyle='round')
    ax.add_collection(line_collection)
    head_marker, = ax.plot(
        [],
        [],
        marker='o',
        linestyle='None',
        markersize=4.0,
        markerfacecolor='#8ec5ff',
        markeredgecolor='none',
        zorder=3,
    )

    base_color = to_rgba('#5ba4f6')
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
        head_marker.set_data([points[-1, 0]], [points[-1, 1]])

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
