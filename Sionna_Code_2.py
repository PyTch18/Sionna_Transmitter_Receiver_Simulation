import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
)

# ------------------------------------------------------------
# 0) LOAD SCENE
# ------------------------------------------------------------
scene_path = "Areas/Pankow/Pankow.xml"
#scene_path = "Areas/Uni/Uni.xml"

scene = load_scene(scene_path)
print("Scene loaded successfully.")
print("Objects:", len(scene.objects))
print("Materials:", len(scene.radio_materials))

# ------------------------------------------------------------
# 1) ADD TX + RXs
# ------------------------------------------------------------
tx_pos = np.array([0.0, 0.0, 10.0])
tx = Transmitter(name="Tx", position=tx_pos.tolist())
scene.add(tx)

num_rx = 15
rx_positions = np.linspace(
    start=[50.0, -20.0, 1.5],
    stop=[350.0, -20.0, 1.5],
    num=num_rx
)

receivers = []
for i, pos in enumerate(rx_positions):
    rx = Receiver(name=f"Rx_{i}", position=pos.tolist())
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

p_solver = PathSolver()

# ------------------------------------------------------------
# 2) RUN SIONNA RT AT 3.5 & 28 GHz
# ------------------------------------------------------------
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

print("Ray tracing finished.")

# ------------------------------------------------------------
# 3) PATH LOSS PER RECEIVER  (using CIR)
# ------------------------------------------------------------
def path_loss_per_rx(paths):
    # a: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps] [web:2]
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    power_rx = np.sum(np.abs(a) ** 2, axis=(1, 2, 3, 4, 5))  # sum over all but rx
    pl_db = -10.0 * np.log10(power_rx + 1e-15)
    return pl_db  # shape [num_rx]


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

# ------------------------------------------------------------
# 4) PDP & RMS DELAY SPREAD FOR ONE RX
# ------------------------------------------------------------
def pdp_for_rx(paths, rx_index=0):
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    a_rx = a[rx_index, 0, 0, 0, :, 0]
    tau_rx = tau[rx_index, 0, :]

    p = np.abs(a_rx) ** 2

    # Keep only valid paths: delay >= 0 and power > 0
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
