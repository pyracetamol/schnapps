import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

st.set_page_config(layout="wide")

k_B = 1  # we use units of J/kT

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

    def __init__(self, lattice_length, J=1, H=0, **kwargs):
        self.lattice_length = lattice_length
        self.J = J
        self.H = H
        self.initialize_lattice(**kwargs)

    @property
    def lattice(self):
        return self._lattice

    @property
    def N(self):
        return self.lattice_length**2

    @property
    def average_energy(self):
        return self.get_average_energy()

    @property
    def average_magnetization(self):
        return self.get_average_magnetization()

    def get_average_energy(self):
        return self.get_total_energy() / self.N

    def get_average_magnetization(self):
        return self.get_total_magnetization() / self.N

    def get_total_energy(self):
        lattice = self.lattice
        L = self.lattice_length
        J, H = self.J, self.H

        neighbors = (
            np.roll(lattice, 1, axis=0) +
            np.roll(lattice, -1, axis=0) +
            np.roll(lattice, 1, axis=1) +
            np.roll(lattice, -1, axis=1)
        )

        interaction_energy = -J * np.sum(lattice * neighbors) / 2
        field_energy = -H * np.sum(lattice)

        return interaction_energy + field_energy

    def get_total_magnetization(self):
        return np.sum(self.lattice)

    def initialize_lattice(self, random=False, spin=1):
        L = self.lattice_length
        if random:
            self._lattice = np.random.choice([-1, 1], size=(L, L))
        elif spin:
            self._lattice = np.full((L, L), spin, dtype=int)
        else:
            raise ValueError("Either lattice is initialized randomly or spin needs to be specified (-1 or 1)")

    def perform_metropolis_step(self, temperature):
        metropolis_step(self._lattice, self.J, self.H, temperature)

    def run_monte_carlo_simulation(self, temperature, number_of_steps=1000, snapshot_interval=50):
        self.initialize_lattice(random=False, spin=1)

        energies = []
        magnetizations = []
        steps = []

        for step in range(number_of_steps):
            self.perform_metropolis_step(temperature)

            if step % snapshot_interval == 0 or step == number_of_steps - 1:
                energies.append(self.average_energy)
                magnetizations.append(self.average_magnetization)
                steps.append(step)

                yield self.lattice.copy(), energies[:], magnetizations[:], steps[:]


# --- Streamlit UI ---
st.title("2D Ising Model Simulator")

L = st.sidebar.slider("Lattice Size", 16, 128, 64, 16)
J = st.sidebar.slider("Exchange Coupling J", 0.1, 2.0, 1.0, 0.1)
H = st.sidebar.slider("External Field H", 0.0, 1.0, 0.0, 0.05)
T = st.sidebar.slider("Temperature T", 0.1, 10.0, 10.0, 0.05)
nsteps = st.sidebar.slider("MC Steps", 100, 5000, 1000, 100)
snapshot_interval = st.sidebar.slider("Snapshot Interval", 1, 100, 5, 5)
run = st.sidebar.button("Run Simulation")

if run:
    with st.spinner("Running simulation..."):
        model = IsingModel2D(L, J, H)

        col1, col2 = st.columns([1, 2])
        with col1:
            spin_placeholder = st.empty()
        with col2:
            chart_placeholder = st.empty()

        chart_data = pd.DataFrame(columns=["Energy", "Magnetization"])

        fig_chart, ax_chart = plt.subplots(figsize=(6, 3))
#      ax_chart.set_title("Energy & Magnetization")
        ax_chart.set_xlabel("Step")
 #       ax_chart.set_ylabel("Value")
        line_energy, = ax_chart.plot([], [], label="Energy", color="royalblue")
        line_magnet, = ax_chart.plot([], [], label="Magnetization", color="crimson")
        ax_chart.legend()
        ax_chart.grid(True)
        chart_plot = chart_placeholder.pyplot(fig_chart)

        all_steps = []
        all_energies = []
        all_magnetizations = []

        for lattice, energies, magnetizations, steps in model.run_monte_carlo_simulation(
            temperature=T, number_of_steps=nsteps, snapshot_interval=snapshot_interval):

            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            ax.imshow(lattice, cmap='plasma', vmin=-1, vmax=1)
 #           ax.set_title(f"Step {steps[-1]}", fontsize=10)
            ax.axis('off')
            fig.tight_layout(pad=0.1)
            spin_placeholder.pyplot(fig)
            plt.close(fig)

            all_steps.append(steps[-1])
            all_energies.append(energies[-1])
            all_magnetizations.append(magnetizations[-1])

            line_energy.set_data(all_steps, all_energies)
            line_magnet.set_data(all_steps, all_magnetizations)
            ax_chart.relim()
            ax_chart.autoscale_view()
            chart_plot.pyplot(fig_chart)
