import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import tempfile
import os
import time

# --- Parameters ---
N = 64
dx = dy = 0.5
snapshot_interval = 100

# --- Sidebar controls ---
st.sidebar.title("Allen-Cahn Parameters")
M = st.sidebar.slider("Mobility (M)", 0.1, 10.0, 5.0)
kappa = st.sidebar.slider("Gradient Energy Coefficient (kappa)", 0.01, 1.0, 0.2)
A = st.sidebar.slider("Free Energy A", 0.1, 5.0, 1.0)
B = st.sidebar.slider("Free Energy B", 0.1, 5.0, 1.0)
ngrains = st.sidebar.slider("Number of Grains", 5, 50, 25)
nsteps = st.sidebar.slider("Number of Steps", 1000, 30000, 2000, step=1000)
run_button = st.sidebar.button("Run Simulation")

# --- Microstructure initialization ---
def procedural_voronoi_smoothed(Nx, Ny, ngrains, sigma, rng_seed):
    np.random.seed(rng_seed)
    points = np.random.rand(ngrains, 2) * [Nx, Ny]
    tree = cKDTree(points)
    x = np.arange(Nx) + 0.5
    y = np.arange(Ny) + 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid_points = np.stack((X.ravel(), Y.ravel()), axis=-1)
    _, grain_ids = tree.query(grid_points)
    grain_map = grain_ids.reshape(Nx, Ny)

    etas = np.zeros((ngrains, Nx, Ny))
    for i in range(ngrains):
        etas[i] = (grain_map == i).astype(float)

    etas_smoothed = gaussian_filter(etas, sigma=(0, sigma, sigma))
    etas_smoothed /= np.sum(etas_smoothed, axis=0, keepdims=True)
    return etas_smoothed, np.ones(ngrains, dtype=bool)

# --- Allen-Cahn simulation ---
def allen_cahn_simulation(M, kappa, A, B, ngrains, dt=0.005, nsteps=20000):
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX ** 2 + KY ** 2

    def dfdeta(etas, eta_i):
        sum_eta_sq = np.sum(etas**2, axis=0) - eta_i**2
        return A * (2.0 * B * eta_i * sum_eta_sq + eta_i**3 - eta_i)

    etas, glist = procedural_voronoi_smoothed(N, N, ngrains, sigma=1.5, rng_seed=1234)
    snapshots = []
    area_records = []

    for step in range(1, nsteps + 1):
        for ig in range(ngrains):
            if not glist[ig]:
                continue
            eta = etas[ig]
            dfdeta_real = dfdeta(etas, eta)
            dfdeta_hat = fft2(dfdeta_real)
            eta_hat = fft2(eta)
            eta_hat = (eta_hat - dt * M * dfdeta_hat) / (1 + dt * M * kappa * K2)
            eta = np.clip(ifft2(eta_hat).real, 0.00001, 0.9999)
            etas[ig] = eta
            if np.sum(eta) / (N * N) <= 0.001:
                glist[ig] = False
                etas[ig] = 0.0

        if step % snapshot_interval == 0 or step == 1:
            eta2 = np.sum(etas**2, axis=0)
            snapshots.append(eta2.copy())
            fractions = [np.sum(etas[ig]) / (N * N) if glist[ig] else 0.0 for ig in range(ngrains)]
            record = {'time': step * dt}
            for ig in range(ngrains):
                record[f'grain_{ig+1}'] = fractions[ig]
            area_records.append(record)

    area_df = pd.DataFrame(area_records)
    return snapshots, area_df

# --- Animation ---
def animate_combined(snapshots, area_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.4)

    im = ax1.imshow(snapshots[0], cmap='viridis', vmin=0.0, vmax=1.0)
    ax1.set_title("Microstructure")
    ax1.axis('off')

    lines = []
    grain_cols = [col for col in area_df.columns if col.startswith('grain_')]
    for col in grain_cols:
        (line,) = ax2.plot([], [], label=col)
        lines.append(line)

    ax2.set_xlim(0, area_df['time'].max())
    flat_area = area_df[grain_cols].values
    ax2.set_ylim(0, flat_area.max())
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Area Fraction")
    ax2.set_title("Grain Area Evolution")
    ax2.grid(True, alpha=0.3)
#    ax2.legend(fontsize="x-small", ncol=2, loc='upper right')

    def update(frame):
        im.set_data(snapshots[frame])
        for i, line in enumerate(lines):
            y = flat_area[:frame + 1, i]
            x = area_df['time'][:frame + 1]
            line.set_data(x, y)
        return [im] + lines

    ani = FuncAnimation(fig, update, frames=len(snapshots), interval=200)
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    ani.save(tmpfile.name, writer='pillow', fps=10)
    plt.close()
    return tmpfile.name

# --- Main ---
st.title("Allen-Cahn: Microstructure and Grain Area Evolution")

if run_button:
    with st.spinner("Running simulation..."):
        snapshots, area_df = allen_cahn_simulation(M, kappa, A, B, ngrains, nsteps=nsteps)
        gif_path = animate_combined(snapshots, area_df)
        st.image(gif_path, use_container_width=True)
        os.unlink(gif_path)
