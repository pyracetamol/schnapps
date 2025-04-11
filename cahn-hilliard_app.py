# cahn_hilliard_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import pandas as pd

# --- App title and layout ---
st.set_page_config(layout="wide")
st.title("Spinodal Decomposition")

# --- Sidebar controls ---
st.sidebar.header("Simulation Parameters")
c0 = st.sidebar.slider("Initial concentration", 0.0, 1.0, 0.5, 0.05)
M = st.sidebar.slider("Mobility", 0.1, 2.0, 0.5, 0.1)
kappa = st.sidebar.slider("Gradient coefficient", 0.1, 2.0, 0.6, 0.1)
W = st.sidebar.slider("Free energy strength (W)", 0.1, 2.0, 1.0, 0.1)
noise = st.sidebar.slider("Noise amplitude", 0.0, 0.05, 0.002, 0.001)
Nsteps = st.sidebar.slider("Number of Time Steps", 500, 10000, 3000, 500)
snapshot_interval = 100
rng_seed = 12345

# --- Fixed parameters ---
N = 64
dx = dy = 1.0
dt = 0.1
L = N * dx

# --- Run Simulation Button ---
run_sim = st.sidebar.button("Run Simulation")

# --- Layout placeholders ---
col1, col2 = st.columns([1, 2])
micro_placeholder = col1.empty()
energy_plot_placeholder = col2.empty()

if run_sim:
    # --- Initialization ---
    rng = np.random.default_rng(rng_seed)
    c = np.empty((Nsteps, N, N), dtype=np.float32)
    c[0] = c0 + noise * rng.standard_normal(size=(N, N))

    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    kmax_dealias = kx.max() * 2.0 / 3.0
    dealias = (np.abs(KX) < kmax_dealias) & (np.abs(KY) < kmax_dealias)

    def dfdc(c):
        return 2 * W * (c * (1 - c)**2 - (1 - c) * c**2)

    def compute_energy(c):
        f_bulk = W * c**2 * (1 - c)**2
        grad_c_x = np.gradient(c, dx, axis=0)
        grad_c_y = np.gradient(c, dy, axis=1)
        grad_energy = 0.5 * kappa * (grad_c_x**2 + grad_c_y**2)
        return np.sum(f_bulk + grad_energy) * dx * dy

    c_hat = fft2(c[0])
    dfdc_hat = np.empty((N, N), dtype=np.complex64)
    energy_records = []

    chart_data = pd.DataFrame(columns=["time", "energy"])

    energy_chart = energy_plot_placeholder.line_chart(chart_data, x="time", y="energy")

    for i in range(Nsteps):
        if i > 0:
            dfdc_hat[:] = fft2(dfdc(c[i - 1]))
            dfdc_hat *= dealias
            c_hat[:] = (c_hat - dt * K2 * M * dfdc_hat) / (1 + dt * M * kappa * K2**2)
            c[i] = ifft2(c_hat).real

        if i % snapshot_interval == 0 or i == 0:
            # --- Microstructure ---
            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            ax.imshow(c[i], cmap='plasma', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.tight_layout(pad=0.2)
            micro_placeholder.pyplot(fig)
            plt.close(fig)

            # --- Energy update ---
            # Initialize figure once before the loop (only once!)
            if i == 0:
                fig_chart, ax_chart = plt.subplots(figsize=(6, 3))
                ax_chart.set_xlabel("Time")
                ax_chart.set_ylabel("Free Energy (normalized)")
                ax_chart.grid(True)
                line_energy, = ax_chart.plot([], [], label="Energy", color="royalblue")
                ax_chart.legend()
                chart_plot = energy_plot_placeholder.pyplot(fig_chart)

            # In the loop, at each update:
            energy = compute_energy(c[i]) / N**2
            new_row = pd.DataFrame({"time": [i * dt], "energy": [energy]})
            chart_data = pd.concat([chart_data, new_row], ignore_index=True)

            line_energy.set_data(chart_data["time"], chart_data["energy"])
            ax_chart.relim()
            ax_chart.autoscale_view()
            chart_plot.pyplot(fig_chart)

