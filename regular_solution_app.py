import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.constants import R

st.set_page_config(page_title="Regular Solution Model Explorer", layout="centered")

st.title("Reguläre Lösung")

st.markdown(
    r"""
Diese App visualisiert die Thermodynamik einer binären Legierung mithilfe des **Modells der regulären Lösungs**.
Sie können erkunden, wie der **Wechselwirkungsparameter** ($\Omega$) und die **Temperatur** ($T$) folgende Größen beeinflussen:
- Gibbssche Energie ($G$)
- Mischungsenthalpie ($H_{\mathrm{mix}}$)
- Entropieterm ($TS_{\mathrm{mix}}$)

Für welche Wahl der Parameter ergibt sich das Verhalten einer **idealen Lösungs**?
### Schlüssel-Gleichungen 
- **Gibbssche Energie**: $G = H_{\mathrm{mix}} + TS_{\mathrm{mix}}$
- **Mischungsenthalpie**: $H_{\mathrm{mix}} = \Omega X_A X_B$
- **Mischungsentropie**: $S_{\mathrm{mix}} = R (X_A \ln X_A + X_B \ln X_B)$
"""
)

# Sidebar parameters
st.sidebar.header("Modell-Parameter")
Omega = st.sidebar.slider("Ω (J/mol)", min_value=-10000, max_value=20000, step=500, value=20000)
T = st.sidebar.slider("Temperatur T (K)", min_value=100, max_value=2000, step=100, value=800)

# Composition range
X_A = np.linspace(0.001, 0.999, 500)
X_B = 1 - X_A

# Thermodynamic terms
S_mix = R * (X_A * np.log(X_A) + X_B * np.log(X_B))
H_mix = Omega * X_A * X_B
G = H_mix + T * S_mix  # assuming mu_A0 = mu_B0 = 0

# Plot
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(X_A, G, label="Gibbssche Energie $G$", lw=2)
ax.plot(X_A, H_mix, label="Mischungsenthalpie $H_{mix}$", ls="--")
ax.plot(X_A, T * S_mix, label="Entropieterm $TS_{mix}$", ls=":")
ax.set_xlabel("Molbruch von A ($X_A$)")
ax.set_ylabel("Energie [J/mol]")
ax.set_title(f"Ω = {Omega} J/mol, T = {T} K")
ax.grid(True)
ax.legend()
st.pyplot(fig)