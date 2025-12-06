# python
# Compatibility shim for Sionna imports — paste into `Sionna_Code.py`

import importlib
import tensorflow as tf

# Known location from your scan
try:
    from sionna.phy.channel.utils import cir_to_time_channel
except Exception:
    cir_to_time_channel = None

# Rayleigh fading class (if present)
try:
    from sionna.phy.channel.rayleigh_block_fading import RayleighBlockFading
except Exception:
    RayleighBlockFading = None

# Try a list of candidates for a "Simulation"-like or ray-tracing provider
SIMULATION_CANDIDATES = [
    ("sionna.phy.channel.tr38901.rays", "RaysGenerator"),
    ("sionna.phy.channel.tr38901.rays", "Rays"),
    ("sionna.rt", "spawn_ray_from_sources"),
    ("sionna.rt.utils.ray_tracing", "spawn_ray_from_sources"),
    ("sionna.rt.antenna_array", "AntennaArray"),
    ("sionna.phy.channel.rayleigh_block_fading", "RayleighBlockFading"),
]

SimulationLike = None
_found_sim = None
for mod_name, attr in SIMULATION_CANDIDATES:
    try:
        m = importlib.import_module(mod_name)
        if hasattr(m, attr):
            SimulationLike = getattr(m, attr)
            _found_sim = f"{mod_name}.{attr}"
            break
    except Exception:
        continue

# Minimal compute_path_loss fallback (keeps code running if Sionna lacks it)
def compute_path_loss(rx_paths, frequency=None):
    """
    Simple fallback: sum path amplitude powers.
    Accepts either an object with `.paths` or an iterable of path-like objects.
    Each path should expose attribute `a` (complex amplitude) or be a complex tensor.
    Returns tf.float32 scalar power.
    """
    paths = getattr(rx_paths, "paths", rx_paths)
    total = tf.constant(0.0, dtype=tf.float32)
    for p in paths:
        # path may be a complex tensor or an object with .a
        a = p if (hasattr(p, "dtype") and p.dtype.is_complex) else getattr(p, "a", None)
        if a is None:
            continue
        a_t = tf.convert_to_tensor(a, dtype=tf.complex64)
        power = tf.math.abs(a_t) ** 2
        total = total + tf.cast(tf.reduce_sum(power), tf.float32)
    return total

# Expose what we found (useful for debug/logging)
_found = {
    "cir_to_time_channel": bool(cir_to_time_channel),
    "RayleighBlockFading": bool(RayleighBlockFading),
    "SimulationLike": _found_sim or None,
}

# Optional: print or log `_found` during module import to confirm what was resolved.
print("Sionna import shim resolved:", _found)
