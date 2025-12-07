import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sionna_vispy
import mitsuba as mi

from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
    Camera,
    cpx_abs_square,
)

def filter_strong_paths(paths, threshold_db=-90.0):
    """
    Keep only strong paths for visualization.
    Works regardless of the exact shape of paths.a.
    """
    a_amp = cpx_abs_square(paths.a)      # |a|^2, same shape as paths.a
    nd = len(a_amp.shape)

    if nd <= 3:
        # Already [num_rx, num_tx, num_paths]
        power = a_amp
    else:
        # Keep first 3 dims (rx, tx, path), sum the rest (antennas, time, ...)
        axes = tuple(range(3, nd))
        power = tf.reduce_sum(a_amp, axis=axes)

    print("power shape:", power.shape, "valid shape:", paths.valid.shape)

    power_db = 10.0 * np.log10(power + 1e-30)
    keep = power_db > threshold_db        # [num_rx,num_tx,num_paths]

    paths._valid = tf.logical_and(paths.valid, keep)
    return paths

scene_path = "Areas/Pankow/Pankow.xml"
scene = load_scene(scene_path)
print("Scene loaded successfully.")
print("Objects:", len(scene.objects))
print("Materials:", len(scene.radio_materials))

def scene_bounds(scene):
    """Return overall [xmin,xmax], [ymin,ymax], [zmin,zmax] of all meshes."""
    shapes = scene._scene.shapes()   # low-level Mitsuba scene [web:18]
    mins = []
    maxs = []
    for s in shapes:
        if isinstance(s, mi.Mesh):
            bb = s.bbox()           # bounding box [web:454]
            mins.append(np.array([bb.min.x, bb.min.y, bb.min.z], float))
            maxs.append(np.array([bb.max.x, bb.max.y, bb.max.z], float))
    mins = np.vstack(mins).min(axis=0)
    maxs = np.vstack(maxs).max(axis=0)
    return mins, maxs

def building_x_bounds(scene, height_thresh=5.0):
    """
    Approximate X-range of buildings, ignoring flat ground plane.
    """
    xs = []
    for s in scene._scene.shapes():
        if not isinstance(s, mi.Mesh):
            continue
        bb = s.bbox()
        h = bb.max.z - bb.min.z
        if h >= height_thresh:   # tall => building, not just ground
            xs.append(bb.min.x)
            xs.append(bb.max.x)
    if not xs:
        raise RuntimeError("No building meshes found; try lowering height_thresh.")
    return float(min(xs)), float(max(xs))

mins, maxs = scene_bounds(scene)
X_MIN, Y_MIN, Z_MIN = mins
X_MAX, Y_MAX, Z_MAX = maxs
print("Scene bounds:", mins, maxs)

X_BUILD_MIN, X_BUILD_MAX = building_x_bounds(scene, height_thresh=5.0)
print("Building X-range:", X_BUILD_MIN, X_BUILD_MAX)

def clamp_to_bounds(pos, mins, maxs, margin=1.0):
    x, y, z = pos
    x = float(np.clip(x, mins[0] + margin, maxs[0] - margin))
    y = float(np.clip(y, mins[1] + margin, maxs[1] - margin))
    z = float(np.clip(z, mins[2] + margin, maxs[2] - margin))
    return [x, y, z]

# Tx roughly above the middle of the building range
tx_x_center = 0.5 * (X_BUILD_MIN + X_BUILD_MAX)
tx_pos = clamp_to_bounds([tx_x_center, 0.0, 10.0], mins, maxs)
tx_pos = np.array(tx_pos, dtype=float)
tx = Transmitter(name="Tx", position=tx_pos.tolist())
scene.add(tx)

# Receivers along the street, limited to building X-range
num_rx = 15
margin_x = 5.0
start_x = X_BUILD_MIN + margin_x
stop_x  = X_BUILD_MAX - margin_x
y_rx = -20.0
z_rx = 1.5

rx_positions = np.linspace(
    start=[start_x, y_rx, z_rx],
    stop=[stop_x,  y_rx, z_rx],
    num=num_rx
)

receivers = []
for i, pos in enumerate(rx_positions):
    pos_clamped = clamp_to_bounds(pos, mins, maxs)
    rx = Receiver(name=f"Rx_{i}", position=pos_clamped)
    scene.add(rx)
    receivers.append(rx)

print("Added receivers:", len(receivers))

scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)
scene.rx_array = scene.tx_array

#3D preview of scene + devices (no rays yet)
scene.preview(show_devices=True)

p_solver = PathSolver()

frequencies = [3.5e9, 28e9]
results = {}

for f in frequencies:
    print(f"\nRunning RT at {f/1e9:.1f} GHz ...")
    scene.frequency = f
    paths = p_solver(
        scene=scene,
        max_depth=3,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=True,
        edge_diffraction=False,
        synthetic_array=True,
    )
    results[f] = paths

    # Interactive preview of rays
    scene.preview(
        paths=paths,
        show_devices=True,
    )

    # Camera: top-down centered on Rx line
    rx_mid = 0.5 * (rx_positions[0] + rx_positions[-1])
    cam = Camera(
        position=[float(rx_mid[0]), float(rx_mid[1]), 250.0],
        look_at=[float(rx_mid[0]), float(rx_mid[1]), 0.0],
    )

    # Filter weak rays for clearer figure
    paths_vis = filter_strong_paths(paths, threshold_db=-90.0)

    scene.render_to_file(
        camera=cam,
        filename=f"pankow_rays_{int(f / 1e9)}GHz.png",
        paths=paths_vis,
        show_devices=True,
        num_samples=256,
        resolution=[1920, 1080],
    )

print("Ray tracing finished.")

def path_loss_per_rx(paths):
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    power_rx = np.sum(np.abs(a) ** 2, axis=(1, 2, 3, 4, 5))
    pl_db = -10.0 * np.log10(power_rx + 1e-15)
    return pl_db

def plot_path_loss(freq, paths):
    pl_db = path_loss_per_rx(paths)
    d = np.linalg.norm(rx_positions - tx_pos[None, :], axis=1)

    plt.figure()
    plt.plot(d, pl_db, "o-")
    plt.title(f"Path Loss vs Distance @ {freq/1e9:.1f} GHz")
    plt.xlabel("Distance Tx–Rx (m)")
    plt.ylabel("Path loss (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for f in frequencies:
    plot_path_loss(f, results[f])

def pdp_for_rx(paths, rx_index=0):
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    a_rx = a[rx_index, 0, 0, 0, :, 0]
    tau_rx = tau[rx_index, 0, :]

    p = np.abs(a_rx) ** 2
    mask = (tau_rx >= 0.0) & (p > 0.0)
    tau_rx = tau_rx[mask]
    p = p[mask]

    idx = np.argsort(tau_rx)
    return tau_rx[idx], p[idx]

def rms_delay_spread(delays, powers):
    p = np.array(powers, dtype=float)
    tau = np.array(delays, dtype=float)
    p_norm = p / (p.sum() + 1e-15)
    tau_mean = np.sum(tau * p_norm)
    return np.sqrt(np.sum(p_norm * (tau - tau_mean) ** 2))

f_test = frequencies[0]
paths_test = results[f_test]
delays, powers = pdp_for_rx(paths_test, rx_index=0)
ds = rms_delay_spread(delays, powers)

plt.figure()
plt.stem(delays * 1e9, powers, basefmt=" ")
plt.title(
    f"PDP for Rx_0 at {f_test/1e9:.1f} GHz\n"
    f"RMS delay spread = {ds*1e9:.2f} ns"
)
plt.xlabel("Delay (ns)")
plt.ylabel("Power")
plt.grid(True)
plt.tight_layout()
plt.show()

print("All processing complete.")
