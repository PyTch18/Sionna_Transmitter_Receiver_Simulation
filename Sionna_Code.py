import numpy as np
import matplotlib.pyplot as plt
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray

# 1) Load Mitsuba/Sionna scene
scene = load_scene("Areas/Pankow/Pankow.xml")   # update path

# 2) Define Tx and Rx
# Example Tx on rooftop (x,y,z in meters, match your scene coordinates)
tx = Transmitter(
    name="tx",
    position=[0.0, 0.0, 25.0],      # change to a real building roof
    carrier_frequency=3.5e9
)
scene.add(tx)

# Rx along a street / grid
num_rx = 20                         # between 5 and 20
x = np.linspace(-80, 80, num_rx)    # adapt to your area
y = np.zeros_like(x)
z = np.ones_like(x) * 1.5           # 1.5 m user height
rx_positions = np.stack([x, y, z], axis=-1)

rx = Receiver(
    name="rx",
    positions=rx_positions
)
scene.add(rx)

# 3) Ray tracing configuration
scene.max_depth = 3                 # >=3 reflections
scene.enable_diffraction(True)
scene.keep_interactions(True)       # to inspect reflections/diffractions

# 4) Run RT at 3.5 GHz
paths_35 = scene.compute_paths()
# Switch to 28 GHz
tx.carrier_frequency = 28e9
paths_28 = scene.compute_paths()

# 5) Channel impulse responses & path loss
def compute_pl_and_cir(paths):
    # distance Tx–Rx
    d = np.linalg.norm(
        paths.rx_positions - paths.tx_positions, axis=-1
    )  # [num_rx]

    # total received power per Rx
    p_rx = np.sum(paths.powers, axis=-1)  # sum over paths

    # path loss in dB (normalize to 1 W Tx power)
    pl_db = -10*np.log10(p_rx + 1e-15)

    return d, pl_db

d_35, pl_35 = compute_pl_and_cir(paths_35)
d_28, pl_28 = compute_pl_and_cir(paths_28)

# 6) PDP & RMS delay spread example for first Rx
def rms_delay_spread(paths, rx_idx=0):
    tau = paths.delays[rx_idx]          # [num_paths]
    p = paths.powers[rx_idx]
    p_norm = p / (np.sum(p) + 1e-15)
    tau_mean = np.sum(tau * p_norm)
    tau_rms = np.sqrt(np.sum(p_norm * (tau - tau_mean)**2))
    return tau, p, tau_rms

tau_35, p_35, rms35 = rms_delay_spread(paths_35, rx_idx=0)
tau_28, p_28, rms28 = rms_delay_spread(paths_28, rx_idx=0)

# 7) Plots for report
plt.figure()
plt.plot(d_35, pl_35, 'o-', label='3.5 GHz')
plt.plot(d_28, pl_28, 'x--', label='28 GHz')
plt.xlabel("Distance [m]")
plt.ylabel("Path loss [dB]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("path_loss_vs_distance.png", dpi=200)

plt.figure()
plt.stem(tau_35*1e9, 10*np.log10(p_35+1e-15), basefmt=" ")
plt.xlabel("Delay [ns]")
plt.ylabel("Power [dB]")
plt.title(f"PDP 3.5 GHz, RMS DS = {rms35*1e9:.2f} ns")
plt.grid(True)
plt.tight_layout()
plt.savefig("pdp_3p5GHz_rx0.png", dpi=200)

plt.figure()
plt.stem(tau_28*1e9, 10*np.log10(p_28+1e-15), basefmt=" ")
plt.xlabel("Delay [ns]")
plt.ylabel("Power [dB]")
plt.title(f"PDP 28 GHz, RMS DS = {rms28*1e9:.2f} ns")
plt.grid(True)
plt.tight_layout()
plt.savefig("pdp_28GHz_rx0.png", dpi=200)
