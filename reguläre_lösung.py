import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.constants import R

st.set_page_config(page_title="Reguläres Lösungsmodell", layout="centered")

st.markdown(
    """
    <div style="position: fixed; top: 60px; right: 20px; font-size: 0.8em; color: gray; z-index: 9999;">
        liebe Grüße von Sabrina ❤︎
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Reguläres Lösungsmodell")

st.markdown(
    """
Diese App visualisiert die Thermodynamik einer binären Legierung mithilfe des **regulären Lösungsmodells**.
Sie können erkunden, wie der **Wechselwirkungsparameter** ($\Omega$) und die **Temperatur** ($T$) folgende Größen beeinflussen:
- Gibbs-Energie ($G$)
- Mischungsenthalpie ($H_{\mathrm{mix}}$)
- Entropieterm ($TS_{\mathrm{mix}}$)

Für welche Parameterwahl ergibt sich das Verhalten der **idealen Lösung**?
### Wichtige Gleichungen
- **Gibbs-Energie**: $G = H_{\mathrm{mix}} + TS_{\mathrm{mix}}$
- **Mischungsenthalpie**: $H_{\mathrm{mix}} = \Omega X_A X_B$
- **Mischungsentropie**: $S_{\mathrm{mix}} = R (X_A \ln X_A + X_B \ln X_B)$
"""
)

# Seitenleiste
st.sidebar.header("Modellparameter")
Omega = st.sidebar.slider("Ω (J/mol)", min_value=-10000, max_value=20000, step=500, value=20000)
T = st.sidebar.slider("Temperatur T (K)", min_value=100, max_value=2000, step=100, value=800)

# Zusammensetzungsbereich
X_A = np.linspace(0.001, 0.999, 500)
X_B = 1 - X_A

# Thermodynamische Größen
S_mix = R * (X_A * np.log(X_A) + X_B * np.log(X_B))
H_mix = Omega * X_A * X_B
G = H_mix + T * S_mix  # assuming mu_A0 = mu_B0 = 0

# Diagramm
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(X_A, G, label="Gibbs-Energie $G$", lw=2)
ax.plot(X_A, H_mix, label="Mischungsenthalpie $H_{mix}$", ls="--")
ax.plot(X_A, T * S_mix, label="Entropieterm $TS_{mix}$", ls=":")
ax.set_xlabel("Molanteil von A ($X_A$)")
ax.set_ylabel("Energie [J/mol]")
ax.set_title(f"Ω = {Omega} J/mol, T = {T} K")
ax.grid(True)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)
