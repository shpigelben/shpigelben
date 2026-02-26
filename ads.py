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

def get_next_collision(P, V, W, H, R):
    """
    Calculates the exact time and normal vector of the next collision
    using continuous ray-casting against 4 lines and 4 circles.
    """
    t_min = float('inf')
    normal = None
    
    # 1. Straight boundaries
    # Right wall (x = W/2)
    if V[0] > 0:
        t = (W/2 - P[0]) / V[0]
        y_hit = P[1] + V[1] * t
        if t > 1e-9 and abs(y_hit) <= H/2 - R:
            if t < t_min: t_min, normal = t, np.array([-1.0, 0.0])
    # Left wall (x = -W/2)
    elif V[0] < 0:
        t = (-W/2 - P[0]) / V[0]
        y_hit = P[1] + V[1] * t
        if t > 1e-9 and abs(y_hit) <= H/2 - R:
            if t < t_min: t_min, normal = t, np.array([1.0, 0.0])
            
    # Top wall (y = H/2)
    if V[1] > 0:
        t = (H/2 - P[1]) / V[1]
        x_hit = P[0] + V[0] * t
        if t > 1e-9 and abs(x_hit) <= W/2 - R:
            if t < t_min: t_min, normal = t, np.array([0.0, -1.0])
    # Bottom wall (y = -H/2)
    elif V[1] < 0:
        t = (-H/2 - P[1]) / V[1]
        x_hit = P[0] + V[0] * t
        if t > 1e-9 and abs(x_hit) <= W/2 - R:
            if t < t_min: t_min, normal = t, np.array([0.0, 1.0])

    # 2. Corner boundaries (quarter circles)
    centers = [
        (W/2 - R, H/2 - R),   # Top-Right
        (-W/2 + R, H/2 - R),  # Top-Left
        (-W/2 + R, -H/2 + R), # Bottom-Left
        (W/2 - R, -H/2 + R)   # Bottom-Right
    ]
    
    v_sq = np.dot(V, V)
    for cx, cy in centers:
        C = np.array([cx, cy])
        Delta = P - C
        b = 2 * np.dot(V, Delta)
        c = np.dot(Delta, Delta) - R**2
        
        discriminant = b**2 - 4 * v_sq * c
        if discriminant >= 0:
            t1 = (-b - np.sqrt(discriminant)) / (2 * v_sq)
            t2 = (-b + np.sqrt(discriminant)) / (2 * v_sq)
            
            for t in (t1, t2):
                if t > 1e-9 and t < t_min:
                    hit = P + V * t
                    # Verify the hit is strictly within the respective corner quadrant
                    if ((cx > 0 and hit[0] >= cx - 1e-6) or (cx < 0 and hit[0] <= cx + 1e-6)) and \
                       ((cy > 0 and hit[1] >= cy - 1e-6) or (cy < 0 and hit[1] <= cy + 1e-6)):
                        t_min = t
                        normal = (C - hit) / R # Pointing inwards
                        normal = normal / np.linalg.norm(normal)

    return t_min, normal

def generate_events(W, H, R, speed, max_time):
    """
    Pre-calculates all collision events up to max_time.
    Returns lists of times, positions, and velocities.
    """
    P = np.array([0.0, 0.0])
    theta = np.radians(37) # Arbitrary non-rational starting angle
    V = np.array([np.cos(theta), np.sin(theta)]) * speed
    
    times = [0.0]
    positions = [P.copy()]
    velocities = [V.copy()]
    
    current_time = 0.0
    while current_time < max_time:
        dt, normal = get_next_collision(P, V, W, H, R)
        if dt == float('inf'):
            break 
            
        P = P + V * dt
        V = V - 2 * np.dot(V, normal) * normal # Specular reflection
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

def draw_rounded_rectangle(ax, W, H, R):
    """
    Plots the table boundary mimicking the GitHub border style.
    """
    t = np.linspace(0, np.pi/2, 50)
    
    tr = np.column_stack((W/2 - R + R*np.cos(t), H/2 - R + R*np.sin(t)))
    tl = np.column_stack((-W/2 + R + R*np.cos(t + np.pi/2), H/2 - R + R*np.sin(t + np.pi/2)))
    bl = np.column_stack((-W/2 + R + R*np.cos(t + np.pi), -H/2 + R + R*np.sin(t + np.pi)))
    br = np.column_stack((W/2 - R + R*np.cos(t + 3*np.pi/2), -H/2 + R + R*np.sin(t + 3*np.pi/2)))
    
    boundary = np.vstack((tr, tl, bl, br, tr[0]))
    ax.plot(boundary[:, 0], boundary[:, 1], color='#d0d7de', linewidth=2.5)

def main():
    parser = argparse.ArgumentParser(description="Simulate 2D dynamical billiard.")
    parser.add_argument("--frames", type=int, default=1500, help="Total number of frames.")
    parser.add_argument("--fps", type=int, default=30, help="Animation frames per second.")
    parser.add_argument("--speed", type=float, default=6.0, help="Particle speed in units per second.")
    parser.add_argument("--width", type=float, default=16.0, help="Table width.")
    parser.add_argument("--height", type=float, default=7.0, help="Table height.")
    parser.add_argument("--radius", type=float, default=1.5, help="Corner radius.")
    parser.add_argument("--trail", type=float, default=1.2, help="Tail length in seconds.")
    parser.add_argument("--output", type=str, default="images/dynamical_billiard_v2.gif", help="Output path.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dt = 1.0 / args.fps
    max_time = args.frames * dt
    evt_times, evt_pos, evt_vel = generate_events(args.width, args.height, args.radius, args.speed, max_time + args.trail)

    fig, ax = plt.subplots(figsize=(args.width/2, args.height/2), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.axis('off')
    
    ax.set_xlim(-args.width/2 - 0.5, args.width/2 + 0.5)
    ax.set_ylim(-args.height/2 - 0.5, args.height/2 + 0.5)
    
    draw_rounded_rectangle(ax, args.width, args.height, args.radius)

    trail_resolution = int(args.trail * args.fps * 2) 
    line_collection = LineCollection([], linewidths=1.5, capstyle='round', joinstyle='round')
    ax.add_collection(line_collection)

    base_color = to_rgba('#0969da')
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