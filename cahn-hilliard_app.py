import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft2, ifft2
from PIL import Image
import tempfile
import os
import pandas as pd

# --- Parameters ---
N = 64
dx = dy = 1.0
L = N * dx
dt = 0.1
snapshot_interval = 50

# --- Sidebar controls ---
st.sidebar.title("Cahn-Hilliard Parameters")
c0 = st.sidebar.slider("Initial concentration", 0.0, 1.0, 0.5, 0.05)
M = st.sidebar.slider("Mobility", 0.1, 2.0, 0.5, 0.1)
kappa = st.sidebar.slider("Gradient coefficient", 0.1, 2.0, 0.6, 0.1)
W = st.sidebar.slider("Free energy strength (W)", 0.1, 2.0, 1.0, 0.1)
noise = st.sidebar.slider("Noise amplitude", 0.0, 0.5, 0.005, 0.005)
Nsteps = st.sidebar.slider("Number of Time Steps", 500, 10000, 5000, 500)

# --- Run simulation button ---
run_simulation = st.sidebar.button("Run Simulation")

if run_simulation:
    with st.spinner("Running simulation..."):

        rng_seed = 12345
        rng = np.random.default_rng(rng_seed)
        c = np.empty((Nsteps, N, N), dtype=np.float32)
        c[0] = c0 + noise * rng.standard_normal(size=(N, N))

        kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2
        kmax_dealias = kx.max() * 2.0 / 3.0
        dealias = (np.abs(KX) < kmax_dealias) & (np.abs(KY) < kmax_dealias)

        def dfdc(c): return 2 * W * (c * (1 - c)**2 - (1 - c) * c**2)

        def compute_energy(c):
            f_bulk = W * c**2 * (1 - c)**2
            grad_c_x = np.gradient(c, dx, axis=0)
            grad_c_y = np.gradient(c, dy, axis=1)
            grad_energy = 0.5 * kappa * (grad_c_x**2 + grad_c_y**2)
            return np.sum(f_bulk + grad_energy) * dx * dy

        c_hat = fft2(c[0])
        dfdc_hat = np.empty((N, N), dtype=np.complex64)
        snapshots = []
        energy_records = []

        for i in range(Nsteps):
            if i > 0:
                dfdc_hat[:] = fft2(dfdc(c[i - 1]))
                dfdc_hat *= dealias
                c_hat[:] = (c_hat - dt * K2 * M * dfdc_hat) / (1 + dt * M * kappa * K2**2)
                c[i] = ifft2(c_hat).real

            if i % snapshot_interval == 0 or i == 0:
                snapshots.append(c[i].copy())
                energy = compute_energy(c[i])
                energy_records.append({'step': i, 'time': i * dt, 'energy': energy})

        # Energy DataFrame (sync length)
        energy_df = pd.DataFrame(energy_records)
        energy_df = energy_df.iloc[:len(snapshots)]
        normalized_energy = energy_df["energy"] / N**2

        # === Combined Animation: Microstructure + Energy Plot ===
        st.markdown("### Spinodal Decomposition: Microstructure and Energy Evolution")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.subplots_adjust(wspace=0.5)

        im = ax1.imshow(snapshots[0], cmap='plasma', vmin=0.0, vmax=1.0)
        ax1.set_title("Microstructure")
        ax1.axis('off')

        line, = ax2.plot([], [], lw=2)
        ax2.set_xlim(0, energy_df["time"].max())
        ax2.set_ylim(normalized_energy.min() * 0.95, normalized_energy.max() * 1.05)
        ax2.set_title("Free Energy")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Energy (normalized)")
        ax2.grid(True, alpha=0.3)

        def animate_combined(i):
            im.set_data(snapshots[i])
            line.set_data(energy_df["time"][:i+1], normalized_energy[:i+1])
            return [im, line]

        ani = FuncAnimation(fig, animate_combined, frames=len(snapshots), interval=200)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        ani.save(tmpfile.name, writer='pillow', fps=10)
        plt.close()

        st.image(tmpfile.name, caption="Cahn–Hilliard: Microstructure and Energy Evolution", use_container_width=True)
        os.unlink(tmpfile.name)
