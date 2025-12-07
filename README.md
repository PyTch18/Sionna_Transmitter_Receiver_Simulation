# Ray-Tracing Based Wireless Channel Modeling in Pankow (Sionna RT)

This project implements a geometry-based wireless channel simulation in a realistic 3D model of the Pankow area in Berlin using NVIDIA Sionna RT. It evaluates propagation at 3.5 GHz and 28 GHz with a single transmitter and multiple receivers along a street canyon, and extracts channel metrics such as path loss, power delay profile (PDP), and RMS delay spread.

---

## Project Overview

- **Environment:** Pankow outdoor urban scene with buildings and street canyon from a Sionna/Mitsuba XML scene.
- **Tools/Libraries:**
  - Python 3.11
  - TensorFlow + Sionna RT
  - Mitsuba (geometry backend)
  - Matplotlib, NumPy
- **Frequencies:** 3.5 GHz and 28 GHz.
- **Nodes:**
  - 1 transmitter (Tx) at ~10 m height.
  - 15 receivers (Rx) along a street line at ~1.5 m height.

The simulation uses deterministic ray tracing with line-of-sight, reflections, refraction, and diffraction enabled to approximate realistic 5G-like channels in an urban canyon.

---

## Repository Structure
```
.
├── Sionna_Code.py                # Full reproducible Python script for RT + analysis
├── Interactive_Simulation.ipynb  # Jupyter notebook with interactive 3D previews
├── Areas/
│   └── Pankow/
│       └── Pankow.xml            # 3D scene geometry
├── figures/
│   ├── 3.5GHz_Lp.jpg             # Path loss vs distance @ 3.5 GHz
│   ├── 2.8GHz_Lp.jpg             # Path loss vs distance @ 28 GHz
│   ├── PDP.jpg                   # Power delay profile + RMS delay spread
│   ├── pankow_rays_3GHz.jpg      # 3D ray visualization @ 3.5 GHz
│   └── pankow_rays_28GHz.jpg     # 3D ray visualization @ 28 GHz
├── Project_Description.pdf       # Original university project brief
└── README.md
```
---

## Features

- **Automatic scene bounds:**
  Uses Mitsuba to compute global scene bounds and the building corridor so that all Tx/Rx positions stay inside the visible area, not outside the ground plane.

- **Configurable layout and parameters:**
  Number of receivers, Tx height, Rx line, margins, and frequency list are adjustable via a small configuration section at the top of the script/notebook.

- **Ray-tracing with Sionna RT:**
  Uses `PathSolver` to compute multipath components (amplitudes and delays) for Tx–Rx pairs at 3.5 and 28 GHz. Reflection, refraction, and diffraction are enabled with a configurable path depth.

- **Channel metrics:**
  From the channel impulse response (CIR), the code computes:
  - Path loss per receiver vs Tx–Rx distance.
  - PDP and RMS delay spread for a selected receiver.

- **Visualization:**
  - 2D plots for path loss and PDP.
  - Static 3D renders of rays in the Pankow scene.
  - Interactive 3D ray viewer in the notebook via `scene.preview(...)`.

---

## Getting Started

### 1. Prerequisites

- Python 3.11
- A virtual environment (recommended)
- CUDA-capable GPU if you want reasonable ray-tracing performance

### 2. Install dependencies

Create and activate venv (optional)
python -m venv .venv

Windows:
```
.venv\Scripts\activate
```
Linux/macOS:
```
source .venv/bin/activate
```
Install core dependencies
```
pip install tensorflow sionna mitsuba matplotlib numpy jupyter
```
Make sure the versions of Sionna RT and Mitsuba you install are compatible.

### 3. Scene files

If you use a different path, update `scene_path` in `Sionna_Code.py` and `Interactive_Simulation.ipynb`.

---

## How to Run

### Option A: Full script

python Sionna_Code.py


This will:

- Load the Pankow scene.
- Place Tx/Rx inside the building corridor.
- Run Sionna RT at 3.5 and 28 GHz.
- Save:
  - Ray images: `pankow_rays_3GHz.png`, `pankow_rays_28GHz.png`.
  - Path-loss plots and the PDP plot (shown via Matplotlib, and optionally saved).

### Option B: Interactive notebook


In the notebook:

1. Run cells in order: imports → configuration → helper functions → scene loading → Tx/Rx placement → ray tracing → plots.
2. Use the cells that call `scene.preview(...)` to open interactive 3D widgets with rays.
3. Adjust parameters (e.g., `num_rx`, `frequencies`) in the configuration cell and re-run downstream cells to compare scenarios.

---

## Example Outputs

- Path loss vs distance at 3.5 GHz and 28 GHz.
- 3D ray visualizations for both carrier frequencies.
- PDP and RMS delay spread for a selected receiver.

These can be reused in reports or presentations illustrating the simulation results.

---

## Authors

- Amr Mohsen  

Developed as part of the **Mobile Communication Networks / Wireless Communication** course at GIU Berlin.

---

## License

Note: The Pankow scene files may be subject to NVIDIA/Sionna licensing terms and are not redistributed in this repository. Please obtain them from the official Sionna RT distribution or generate your own scene.
