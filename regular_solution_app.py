import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.constants import R

st.set_page_config(page_title="Regular Solution Model Explorer", layout="centered")

st.title("Regular Solution Model")

st.markdown(
    """
This app visualizes the thermodynamics of a binary alloy using the **regular solution model**.
You can explore how the **interaction parameter** ($\Omega$) and **temperature** ($T$) affect:
- Gibbs free energy ($G$)
- Enthalpy of mixing ($H_{\mathrm{mix}}$)
- Entropy term ($TS_{\mathrm{mix}}$)

For which choice of parameters do you recover the **ideal solution** behavior?
### Key Equations
- **Gibbs Free Energy**: $G = H_{\mathrm{mix}} + TS_{\mathrm{mix}}$
- **Enthalpy of Mixing**: $H_{\mathrm{mix}} = \Omega X_A X_B$
- **Entropy of Mixing**: $S_{\mathrm{mix}} = R (X_A \ln X_A + X_B \ln X_B)$
"""
)

# Sidebar parameters
st.sidebar.header("Model Parameters")
Omega = st.sidebar.slider("Ω (J/mol)", min_value=-10000, max_value=20000, step=500, value=5000)
T = st.sidebar.slider("Temperature T (K)", min_value=100, max_value=2000, step=100, value=800)

# Composition range
X_A = np.linspace(0.001, 0.999, 500)
X_B = 1 - X_A

# Thermodynamic terms
S_mix = R * (X_A * np.log(X_A) + X_B * np.log(X_B))
H_mix = Omega * X_A * X_B
G = H_mix + T * S_mix  # assuming mu_A0 = mu_B0 = 0

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(X_A, G, label="Gibbs Free Energy $G$", lw=2)
ax.plot(X_A, H_mix, label="Enthalpy of Mixing $H_{mix}$", ls="--")
ax.plot(X_A, T * S_mix, label="Entropy Term $TS_{mix}$", ls=":")
ax.set_xlabel("Mole fraction of A ($X_A$)")
ax.set_ylabel("Energy [J/mol]")
ax.set_title(f"Ω = {Omega} J/mol, T = {T} K")
ax.grid(True)
ax.legend()
st.pyplot(fig)