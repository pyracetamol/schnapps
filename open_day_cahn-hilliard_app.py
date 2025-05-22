# cahn_hilliard_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

# --- App title and layout ---
st.set_page_config(layout="wide")
st.title("Cahn-Hilliard: spinodale Entmischung")

# --- Sidebar controls ---
st.sidebar.header("Simulations-Parameter")
c0 = st.sidebar.slider("Anfängliche Konzentration", 0.1, 0.9, 0.5, 0.05)
M = st.sidebar.slider("Mobilität", 0.1, 2.0, 0.5, 0.1)
kappa = st.sidebar.slider("Gradienten-Koeffizient", 0.1, 2.0, 0.6, 0.1)
W = st.sidebar.slider("Stärke der freien Energie (W)", 0.1, 2.0, 1.0, 0.1)
noise = st.sidebar.slider("Rausch-Amplitude", 0.0, 0.5, 0.5, 0.05)
Nsteps = st.sidebar.slider("Anzahl der Schritte", 500, 50000, 25000, 500)
snapshot_interval = 100
rng_seed = 12345

# --- Fixed parameters ---
N = 64
dx = dy = 1.0
dt = 0.1
L = N * dx

# --- Run Simulation Button ---
run_sim = st.sidebar.button("Simulation starten")
st.sidebar.page_link("http://localhost:8000/index.html", label="Übersicht", icon="🏠")

# --- Layout placeholders ---
col1, col2 = st.columns([1, 2])
micro_placeholder = col1.empty()
info_box = col2.empty()

# --- Info text replacing energy plot ---
info_box.markdown(
    """
    **Spinodale Entmischung** ist ein natürlicher Prozess, bei dem sich ein Gemisch spontan in zwei getrennte Bereiche aufteilt.  
    Die spinodale Entmischung spielt eine wichtige Rolle in vielen natürlichen und technischen Systemen.  
    Man erkennt sie zum Beispiel daran, wie sich Öl und Essig in einer Vinaigrette trennen oder wie sich Polymermischungen, Metalle und sogar biologische Membranen in klar abgegrenzte Regionen organisieren.

    #### Wussten Sie schon?

    In **Batterien** kann spinodale Entmischung schleichend langfristige Schäden verursachen.  
    Zu langes oder zu starkes Laden kann dazu führen, dass sich die Materialien im Inneren der Batterie trennen.  
    Diese Phasentrennung führt zu ungleichmäßigen Reaktionen, inneren Spannungen und Rissen - und löst eine Kette von Prozessen aus, die letztlich die Leistung der Batterie verschlechtern.

    Ein besseres Verständnis dieses Phänomens kann helfen, das Verhalten und die Lebensdauer von Materialien besser vorherzusagen!
    """
)

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

    c_hat = fft2(c[0])
    dfdc_hat = np.empty((N, N), dtype=np.complex64)

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