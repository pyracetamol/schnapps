import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from PIL import Image
import tempfile
import os

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
        self.initialize_lattice(random=True)

        snapshots = []
        energies = []
        magnetizations = []
        steps = []

        for step in range(number_of_steps):
            self.perform_metropolis_step(temperature)

            if step % snapshot_interval == 0 or step == number_of_steps - 1:
                snapshots.append(self.lattice.copy())
                energies.append(self.average_energy)
                magnetizations.append(self.average_magnetization)
                steps.append(step)

        return snapshots, energies, magnetizations, steps


# --- Streamlit UI ---
st.title("2D Ising Model Simulator")

L = st.sidebar.slider("Lattice Size", 16, 128, 64, 16)
J = st.sidebar.slider("Exchange Coupling J", 0.1, 2.0, 1.0, 0.1)
H = st.sidebar.slider("External Field H", 0.0, 1.0, 0.0, 0.05)
T = st.sidebar.slider("Temperature T", 0.1, 10.0, 10.0, 0.05)
nsteps = st.sidebar.slider("MC Steps", 100, 5000, 1000, 100)
snapshot_interval = st.sidebar.slider("Snapshot Interval", 1, 100, 5, 5)

if st.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        model = IsingModel2D(L, J, H)
        snapshots, energies, magnetizations, steps = model.run_monte_carlo_simulation(
            temperature=T, number_of_steps=nsteps, snapshot_interval=snapshot_interval
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.subplots_adjust(wspace=0.5)

        im = ax1.imshow(snapshots[0], cmap='plasma', vmin=-1, vmax=1)
        ax1.set_title("Spin Configuration")
        ax1.axis('off')

        line_e, = ax2.plot([], [], label="Energy")
        line_m, = ax2.plot([], [], label="Magnetization")
        ax2.set_xlim(min(steps), max(steps))
        ax2.set_ylim(min(min(energies), min(magnetizations)) * 1.1, max(max(energies), max(magnetizations)) * 1.1)
        ax2.set_title("Energy & Magnetization")
        ax2.set_xlabel("Step")
        ax2.legend()
        ax2.grid(True)

        def animate(i):
            im.set_data(snapshots[i])
            line_e.set_data(steps[:i+1], energies[:i+1])
            line_m.set_data(steps[:i+1], magnetizations[:i+1])
            return [im, line_e, line_m]

        ani = FuncAnimation(fig, animate, frames=len(snapshots), interval=200)
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        ani.save(tmpfile.name, writer='pillow', fps=10)
        plt.close()

        st.image(tmpfile.name, use_container_width=True)
        os.unlink(tmpfile.name)
