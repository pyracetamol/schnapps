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
Ihr könnt erkunden, wie der **Wechselwirkungsparameter** ($\Omega$) und die **Temperatur** ($T$) folgende Größen beeinflussen:
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

# Zweite Ableitung von G nach X_A (analytisch)
# d²G/dX_A² = -2Ω + RT(1/X_A + 1/X_B)
d2G_dX2 = -2 * Omega + R * T * (1 / X_A + 1 / X_B)
spinodal_mask = d2G_dX2 < 0

# Diagramm
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(X_A, G, label="Gibbs-Energie $G$", lw=2)
ax.plot(X_A, H_mix, label="Mischungsenthalpie $H_{mix}$", ls="--")
ax.plot(X_A, T * S_mix, label="Entropieterm $TS_{mix}$", ls=":")

# Spinodale Region einzeichnen
if spinodal_mask.any():
    ax.fill_between(
        X_A, G.min() * 1.1, G.max() * 1.1,
        where=spinodal_mask,
        alpha=0.15, color="red",
        label="Spinodale Zone ($\\partial^2 G/\\partial X_A^2 < 0$)"
    )
    spinodal_transitions = np.where(np.diff(spinodal_mask.astype(int)))[0]
    for idx in spinodal_transitions:
        ax.axvline(X_A[idx], color="red", lw=1, ls=":", alpha=0.6)

ax.set_xlabel("Molanteil von A ($X_A$)")
ax.set_ylabel("Energie [J/mol]")
ax.set_title(f"Ω = {Omega} J/mol, T = {T} K")
ax.grid(True)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

# ── Theorie: Spinodale Entmischung ──────────────────────────────────────────
st.divider()
st.header("Spinodale Entmischung")

st.markdown(
    """
### Was passiert physikalisch?

Stellt euch vor, ihr mischt zwei Metalle A und B bei hoher Temperatur zu einer homogenen
Legierung – alles schön gleichmäßig verteilt. Nun kühlt ihr die Legierung ab.
Je nach Zusammensetzung und Wechselwirkung zwischen den Atomen kann das System instabil
werden und sich **spontan** in zwei Phasen aufteilen: eine A-reiche und eine B-reiche Phase.

Diesen Vorgang nennt man **spinodale Entmischung** (*spinodal decomposition*).
Er unterscheidet sich von der klassischen Keimbildung: Es gibt **keine Energiebarriere**
und **keinen Keim** – die Entmischung beginnt sofort und überall gleichzeitig.

---

### Das Stabilitätskriterium

Das Entscheidende steckt in der **Krümmung** der Gibbs-Energie-Kurve.

> **Eine Phase ist stabil, wenn die Gibbs-Energie-Kurve nach oben gewölbt ist
> (konvex), d.h. wenn gilt:**
> $$\\frac{\\partial^2 G}{\\partial X_A^2} > 0$$

Ist die Kurve dagegen **nach unten gewölbt (konkav)**, also

$$\\frac{\\partial^2 G}{\\partial X_A^2} < 0,$$

dann ist jede kleine Konzentrationsänderung energetisch **günstig** – das System
entmischt sich spontan. Dieser Bereich heißt **spinodale Zone** (rot markiert im Diagramm).

**Intuition:** Wenn die Kurve konkav ist, liegt der Mittelwert zweier benachbarter
Punkte *unterhalb* der Kurve. Das System kann also Energie gewinnen, indem es sich
in zwei Zusammensetzungen aufteilt.

---

### Die zweite Ableitung im regulären Lösungsmodell

Für das reguläre Lösungsmodell lässt sich $\\partial^2 G / \\partial X_A^2$
analytisch berechnen:

$$\\frac{\\partial^2 G}{\\partial X_A^2} = -2\\Omega + RT\\left(\\frac{1}{X_A} + \\frac{1}{X_B}\\right)$$

- Der Term $-2\\Omega$ kommt von der **Mischungsenthalpie** (attraktive/repulsive Wechselwirkung).
- Der Term $RT(1/X_A + 1/X_B)$ kommt von der **Mischungsentropie** und ist immer positiv –
  Entropie stabilisiert die Mischung.

**Spinodale Entmischung tritt auf**, wenn $\\Omega > 0$ (gleichartige Bindungen bevorzugt)
und die Temperatur niedrig genug ist, dass der Enthalpieterm dominiert.

---

### Zum Ausprobieren 

Zieht die Regler und beobachtet die rote Zone:

| Frage | Hinweis |
|---|---|
| Ab welchem $\\Omega$ erscheint die spinodale Zone? | Erhöht $\\Omega$ langsam von 0 |
| Wie verändert höhere Temperatur die Zone? | Entropie wird wichtiger |
| Warum gibt es bei negativem $\\Omega$ keine Entmischung? | Was bedeutet $\\Omega < 0$ physikalisch? |
"""
)
