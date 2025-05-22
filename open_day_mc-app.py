import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

st.set_page_config(layout="wide")

k_B = 1  # units of J/kT

@njit
def metropolis_step(lattice, J, H, T):
    L = lattice.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        S = lattice[i, j]
        neighbors = (
            lattice[(i + 1) % L, j] + lattice[i, (j + 1) % L] +
            lattice[(i - 1) % L, j] + lattice[i, (j - 1) % L]
        )
        dE = 2 * S * (J * neighbors + H)

        if dE < 0 or np.random.rand() < np.exp(-dE / (k_B * T)):
            lattice[i, j] = -S


class IsingModel2D:
    def __init__(self, lattice_length, J=1, H=0):
        self.lattice_length = lattice_length
        self.J = J
        self.H = H
        self._lattice = np.random.choice([-1, 1], size=(lattice_length, lattice_length))

    @property
    def lattice(self):
        return self._lattice

    def perform_metropolis_step(self, T):
        metropolis_step(self._lattice, self.J, self.H, T)

    def run_temperature_sweep(self, T_values, steps_per_temp=100):
        for T in T_values:
            for _ in range(steps_per_temp):
                self.perform_metropolis_step(T)
            yield self.lattice.copy(), T


# --- UI ---
st.title("Monte Carlo: Glücksspiel mit der Physik")

st.sidebar.markdown("### Einstellungen")
L = st.sidebar.slider("Gittergröße", 16, 128, 64, 16)
J = st.sidebar.slider("Wechselwirkung J", 0.1, 2.0, 1.0, 0.1)
H = st.sidebar.slider("Externes Feld H", 0.0, 1.0, 0.0, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### Legende")
st.sidebar.markdown("🟡 = Spin hoch (↑)")  
st.sidebar.markdown("🟣 = Spin runter (↓)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Szenarien")
cool = st.sidebar.button("🧊 System kühlen")
heat = st.sidebar.button("🔥 System erhitzen")
st.sidebar.page_link("http://localhost:8000/index.html", label="Übersicht", icon="🏠")

# --- Layout ---
col1, col2 = st.columns([1, 2])
temp_placeholder = col1.empty()
spin_placeholder = col1.empty()

col2.markdown(
    """
    **Monte-Carlo-Simulationen** nutzen Zufall, um das Verhalten von Systemen zu erkunden.

    Dieses Modell zeigt ein Gitter aus winzigen Magneten, sogenannten **Spins**.  
    Jeder Spin kann **nach oben (↑)** oder **nach unten (↓)** zeigen, und sie „möchten“ sich mit ihren Nachbarn ausrichten, um Energie zu minimieren.

    - Wenn man das System **abkühlt**, beginnen sich die Spins auszurichten – es entstehen große, geordnete Farbflächen.  
    - Wenn man es **erhitzt**, zerstört die thermische Bewegung diese Ordnung – die Farben flackern und vermischen sich, weil der Zufall die Kontrolle übernimmt.

    #### Wussten Sie schon?

    Monte-Carlo-Simulationen werden weit über die Physik hinaus eingesetzt:  
    Sie helfen bei der Modellierung von **magnetischen Materialien**, **chemischen Reaktionen**, **sozialem Verhalten**, **Verkehrsfluss** und sogar von **Finanzmärkten**.
    """
)

# --- Main simulation logic ---
def run_sweep(mode="cool"):
    model = IsingModel2D(L, J, H)
    if mode == "cool":
        T_values = np.linspace(5.0, 1.0, 25)
    elif mode == "heat":
        T_values = np.linspace(1.0, 5.0, 25)
    else:
        return

    for lattice, T in model.run_temperature_sweep(T_values, steps_per_temp=500):
        temp_placeholder.markdown(f"#### Temperatur: {T:.2f}")
        fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
        ax.imshow(lattice, cmap="plasma", vmin=-1, vmax=1)
        ax.axis('off')
        fig.tight_layout(pad=0.1)
        spin_placeholder.pyplot(fig)
        plt.close(fig)

if cool:
    run_sweep("cool")

if heat:
    run_sweep("heat")
