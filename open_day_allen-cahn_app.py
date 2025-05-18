import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

st.set_page_config(layout="wide")

# --- Parameters ---
N = 64
dx = dy = 0.5
snapshot_interval = 100

# --- Sidebar controls ---
st.sidebar.title("Allen-Cahn Parameters")
M = st.sidebar.slider("Mobility (M)", 0.1, 10.0, 10.0)
kappa = st.sidebar.slider("Gradient Energy Coefficient (kappa)", 0.01, 1.0, 0.2)
A = st.sidebar.slider("Free Energy A", 0.1, 5.0, 1.0)
B = st.sidebar.slider("Free Energy B", 0.1, 5.0, 1.0)
ngrains = st.sidebar.slider("Number of Grains", 5, 50, 38)
nsteps = st.sidebar.slider("Number of Steps", 1000, 30000, 2000, step=1000)
snapshot_interval = st.sidebar.slider("Snapshot Interval", 0, 1000, 50, 50)
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

# --- Allen-Cahn simulation with microstructure and education text ---
def allen_cahn_simulation_live(M, kappa, A, B, ngrains, dt=0.005, nsteps=20000, snapshot_interval=100):
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX ** 2 + KY ** 2

    def dfdeta(etas, eta_i):
        sum_eta_sq = np.sum(etas**2, axis=0) - eta_i**2
        return A * (2.0 * B * eta_i * sum_eta_sq + eta_i**3 - eta_i)

    etas, glist = procedural_voronoi_smoothed(N, N, ngrains, sigma=1.5, rng_seed=1234)

    col1, col2 = st.columns([1, 2])
    micro_placeholder = col1.empty()

    col2.markdown(
        """
        Die **Allen-Cahn-Gleichung** beschreibt, wie sich die innere Struktur eines Materials mit der Zeit verändert.  
        Sie modelliert, wie kleine Körner in einem Festkörper allmählich schrumpfen und schließlich verschwinden.

        Dieser Prozess, bekannt als **Kornvergröberung**, tritt in Metallen, Keramiken und sogar in biologischen Geweben auf.  
        Wenn kleinere Körner verschwinden und größere wachsen, können sich die mechanischen Eigenschaften eines Materials – wie Festigkeit, Härte und Sprödigkeit – erheblich verändern.

        #### Wussten Sie schon?

        In **automobilen Katalysatoren** ist die Kornvergröberung eine der Ursachen dafür, dass die Effizienz mit der Zeit und unter Hitzeeinwirkung nachlässt.

        Ein Verständnis dieser Entwicklung hilft Materialwissenschaftler/innen, leistungsfähigere Werkstoffe zu entwickeln - für alles von **Strahltriebwerken** bis hin zu **Mikrochips**.


        """
    )

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
            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            ax.imshow(eta2, cmap='viridis', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.tight_layout(pad=0.2)
            micro_placeholder.pyplot(fig)
            plt.close(fig)

# --- Main ---
st.title("Allen-Cahn: Mikrostrukturevolution")

if run_button:
    allen_cahn_simulation_live(M, kappa, A, B, ngrains, nsteps=nsteps, snapshot_interval=snapshot_interval)