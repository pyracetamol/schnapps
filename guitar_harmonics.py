"""
Guitar String Harmonics Visualizer
Converted from Wolfram Mathematica notebook to Streamlit

Run with:
    pip install streamlit matplotlib numpy pandas
    streamlit run guitar_harmonics.py

Animation runs entirely in the browser via an HTML5 Canvas component,
so there are no Python round-trips per frame and no page-flash.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Computer Modern Sans throughout matplotlib ──────────────────────────────
mpl.rcParams.update({
    "font.family":        "cmss10",
    "axes.unicode_minus": False,   # use ASCII hyphen-minus; cmss10 lacks U+2212
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
})

# ── Physics constants ────────────────────────────────────────────────────────
A          = 0.25
L          = 25.5
T_over_rho = 41.77

def wavelength(n):   return 2 * L / n
def angular_freq(n): return (n * np.pi / L) * np.sqrt(T_over_rho)
def frequency_hz(n): return angular_freq(n) / (2 * np.pi)

harmonic_labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}

FORCED_NODES = {
    2: "12th fret",
    3: "7th fret",
    4: "5th fret",
    5: "just before 4th fret",
}

FRET_POSITIONS = [
    24.069, 22.718, 21.443, 20.239, 19.103,
    18.031, 17.019, 16.064, 15.162, 14.311, 13.508, 12.75,
]

# ── Colours ──────────────────────────────────────────────────────────────────
BG     = "#f5f0e8"
WAVE_C = "#1a237e"
ORANGE = "#e65c00"
ROW_HI = "#e8dfc8"   # highlighted row in table

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Guitar String Harmonics", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'cmss10', 'Latin Modern Sans', 'Source Sans 3', sans-serif;
}
h1, h2, h3, p, label, div, span, .stMarkdown {
    font-family: 'cmss10', 'Latin Modern Sans', 'Source Sans 3', sans-serif !important;
}
button[title="View fullscreen"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("Guitar String Harmonics")
st.markdown("Interactive visualizer for standing waves on a guitar string.")


# ── Helper: render a matplotlib figure into the sidebar with no padding ──────
def sidebar_fig(fig):
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    n = st.selectbox(
        "Harmonic",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: harmonic_labels[x],
        index=0,
    )

    speed = st.selectbox("Animation speed", ["1x", "5x", "10x"], index=0)
    speed_val = {"1x": 1.0, "5x": 5.0, "10x": 10.0}[speed]

    # ── Harmonics table ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Harmonics")

    tbl_rows = [
        (harmonic_labels[i], f"{frequency_hz(i):.2f}", FORCED_NODES.get(i, "open string"))
        for i in range(1, 6)
    ]
    col_headers = ["Harmonic", r"$\nu$ (Hz)", "Fret"]
    col_x       = [0.03, 0.35, 0.62]   # left edge of each column (axes fraction)
    n_rows      = len(tbl_rows)
    row_h       = 1.0 / (n_rows + 1.6) # fractional height per row

    fig_tbl, ax_tbl = plt.subplots(figsize=(2.9, (n_rows + 1.6) * 0.28), facecolor=BG)
    ax_tbl.set_facecolor(BG)
    ax_tbl.axis("off")

    # Header row
    for cx, hdr in zip(col_x, col_headers):
        ax_tbl.text(cx, 1 - 0.7 * row_h, hdr,
                    transform=ax_tbl.transAxes,
                    va="center", ha="left", fontsize=10,
                    fontweight="bold", color="#222")
    # Separator under header (drawn in axes coords via plot)
    sep_y = 1 - 1.25 * row_h
    ax_tbl.plot([0.01, 0.99], [sep_y, sep_y],
                color="#999", linewidth=0.8,
                transform=ax_tbl.transAxes, zorder=1)

    # Data rows
    for r, (h_lbl, nu_str, fret_str) in enumerate(tbl_rows):
        y_ctr = 1 - (1.6 + r + 0.5) * row_h
        # Highlight selected harmonic
        if r + 1 == n:
            rect_y = 1 - (1.6 + r + 1.0) * row_h
            ax_tbl.fill_between([0.01, 0.99], rect_y, rect_y + row_h,
                                color=ROW_HI,
                                transform=ax_tbl.transAxes, zorder=0)
        for cx, cell in zip(col_x, (h_lbl, nu_str, fret_str)):
            ax_tbl.text(cx, y_ctr, cell,
                        transform=ax_tbl.transAxes,
                        va="center", ha="left", fontsize=9.5, color="#222")

    sidebar_fig(fig_tbl)

    # ── Equations ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Equations")

    eq_lines = [
        r"$y = 2A\,\sin\!\left(\dfrac{2\pi x}{\lambda}\right)\cos(\omega t)$",
        r"$\lambda = \dfrac{2L}{n}$",
        r"$\omega = \dfrac{n\pi}{L}\sqrt{\dfrac{T}{\rho}}$",
        r"$\nu = \dfrac{\omega}{2\pi}$",
    ]
    n_eq = len(eq_lines)
    fig_eq, ax_eq = plt.subplots(figsize=(2.9, n_eq * 0.72), facecolor=BG)
    ax_eq.set_facecolor(BG)
    ax_eq.axis("off")
    for k, line in enumerate(eq_lines):
        ax_eq.text(0.06, 1 - (k + 0.5) / n_eq, line,
                   transform=ax_eq.transAxes,
                   va="center", ha="left", fontsize=12, color="#222")
    sidebar_fig(fig_eq)


# ── Derived quantities ────────────────────────────────────────────────────────
lam   = wavelength(n)
omega = angular_freq(n)
nu    = frequency_hz(n)

x_vals = np.linspace(0, L, 500)
y_snap = (2 * A) * np.sin((2 * np.pi * x_vals) / lam)  # t = 0 snapshot

# ── Wave profile figure ───────────────────────────────────────────────────────
fig_wave, ax_wave = plt.subplots(figsize=(6.5, 4.2), facecolor=BG)
ax_wave.set_facecolor(BG)

ax_wave.plot(x_vals, y_snap, color=WAVE_C, linewidth=2.5)
ax_wave.axhline(0, color="#888", linewidth=0.8, linestyle="--")

nodes_x = np.linspace(0, L, n + 1)
ax_wave.plot(nodes_x, np.zeros_like(nodes_x), "o",
             color="#c62828", markersize=7, label="nodes", zorder=5)

antinodes_x = [(nodes_x[i] + nodes_x[i + 1]) / 2 for i in range(len(nodes_x) - 1)]
antinode_y  = [(2 * A) * np.sin((2 * np.pi * xi) / lam) for xi in antinodes_x]
ax_wave.plot(antinodes_x, antinode_y, "^",
             color=ORANGE, markersize=8, label="antinodes", zorder=5)

ax_wave.set_xlabel("Position along string  x")
ax_wave.set_ylabel("Displacement  y")
ax_wave.set_title(f"Standing Wave Profile: {harmonic_labels[n]} harmonic  (t = 0)")
ax_wave.tick_params(colors="#333")
for spine in ax_wave.spines.values():
    spine.set_edgecolor("#bbb")
ax_wave.legend(facecolor=BG, edgecolor="#bbb", labelcolor="#333")
fig_wave.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.13)

# ── Physics recap figure (horizontal, two columns) ────────────────────────────
# Title: "Harmonic: 1st (n = 1)"
# Left col:  L = ...,  λ = ...
# Right col: ω = ...,  ν = ...
fig_recap, ax_rc = plt.subplots(figsize=(6.5, 1.25), facecolor=BG)
ax_rc.set_facecolor(BG)
ax_rc.axis("off")

title_str = f"Harmonic: {harmonic_labels[n]}  (n = {n})"
ax_rc.text(0.5, 0.92, title_str,
           transform=ax_rc.transAxes,
           va="top", ha="center", fontsize=12, fontweight="bold", color="#222")

left_lines  = [
    f"$L = {L}$",
    f"$\\lambda = {lam:.3f}$",
]
right_lines = [
    f"$\\omega = {omega:.3f}$ rad/s",
    f"$\\nu = {nu:.2f}$ Hz",
]

for k, line in enumerate(left_lines):
    ax_rc.text(0.08, 0.62 - k * 0.36, line,
               transform=ax_rc.transAxes,
               va="top", ha="left", fontsize=11, color="#222")

for k, line in enumerate(right_lines):
    ax_rc.text(0.54, 0.62 - k * 0.36, line,
               transform=ax_rc.transAxes,
               va="top", ha="left", fontsize=11, color="#222")

fig_recap.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ── Layout: canvas left, (wave + recap) right ────────────────────────────────
col_canvas, col_plot = st.columns([1, 1])

with col_plot:
    st.pyplot(fig_wave)
    plt.close(fig_wave)
    st.pyplot(fig_recap)
    plt.close(fig_recap)

# ── HTML5 Canvas animation ────────────────────────────────────────────────────
fret_js   = "[" + ",".join(f"{fp}" for fp in FRET_POSITIONS) + "]"
node_frac = 1.0 - 1.0 / n
has_node  = "true" if n > 1 else "false"

canvas_html = f"""
<div style="background:{BG}; padding:8px; border-radius:6px; display:inline-block;">
  <canvas id="fretboard" width="300" height="520"
          style="display:block; background:{BG};"></canvas>
  <div style="text-align:center; margin-top:6px;">
    <button id="playBtn"
      style="font-family:'Source Sans 3',sans-serif; font-size:14px;
             padding:6px 22px; border-radius:4px; border:1px solid #999;
             background:#fff; cursor:pointer;">Play</button>
  </div>
</div>

<script>
(function() {{
  const canvas  = document.getElementById("fretboard");
  const ctx     = canvas.getContext("2d");
  const playBtn = document.getElementById("playBtn");

  const N          = {n};
  const A_amp      = {A};
  const L_len      = {L};
  const lam        = {lam};
  const omega      = {omega};
  const fretY_data = {fret_js};
  const hasNode    = {has_node};
  const nodeFrac   = {node_frac:.6f};

  const W = canvas.width, H = canvas.height;
  const marginTop = 30, marginBot = 30;
  const stringsX  = [50, 90, 130, 170, 210, 250];
  const activeX   = stringsX[N - 1];

  function toCanvasY(wy) {{
    return (H - marginBot) - (wy / L_len) * (H - marginTop - marginBot);
  }}

  const WOOD     = "#c9a84c";
  const FRET_COL = "#7a5c1e";
  const STRING   = "#9e9e9e";
  const WAVE_COL = "#1a237e";
  const ORANGE   = "#e65c00";

  const T_phys = (2 * Math.PI) / omega;
  const SPEED  = {speed_val};
  let t_phys   = 0.0;
  let playing  = false;
  let rafId    = null;
  let lastWall = null;

  function draw() {{
    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = WOOD;
    ctx.fillRect(stringsX[0] - 20, marginTop,
                 stringsX[5] + 20 - (stringsX[0] - 20),
                 H - marginTop - marginBot);

    ctx.strokeStyle = ORANGE;
    ctx.lineWidth   = 5;
    ctx.lineCap     = "square";
    for (const py of [marginTop, H - marginBot]) {{
      ctx.beginPath();
      ctx.moveTo(stringsX[0] - 18, py);
      ctx.lineTo(stringsX[5] + 18, py);
      ctx.stroke();
    }}

    ctx.strokeStyle = FRET_COL;
    ctx.lineWidth   = 1.8;
    for (const fy of fretY_data) {{
      const py = toCanvasY(fy);
      ctx.beginPath();
      ctx.moveTo(stringsX[0] - 18, py);
      ctx.lineTo(stringsX[5] + 18, py);
      ctx.stroke();
    }}

    for (let i = 0; i < 6; i++) {{
      ctx.strokeStyle = STRING;
      ctx.lineWidth   = 0.8 + i * 0.18;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.moveTo(stringsX[i], marginTop);
      ctx.lineTo(stringsX[i], H - marginBot);
      ctx.stroke();
    }}
    ctx.globalAlpha = 1.0;

    ctx.fillStyle = "#222";
    for (const sx of stringsX) {{
      ctx.beginPath();
      ctx.arc(sx, H - marginBot, 5, 0, 2 * Math.PI);
      ctx.fill();
    }}

    const nPts = 300;
    ctx.strokeStyle = WAVE_COL;
    ctx.lineWidth   = 2.8;
    ctx.lineJoin    = "round";
    ctx.beginPath();
    for (let i = 0; i <= nPts; i++) {{
      const x  = (i / nPts) * L_len;
      const y  = 2 * A_amp * Math.sin(2 * Math.PI * x / lam) * Math.cos(omega * t_phys);
      const px = activeX + y * 80;
      const py = toCanvasY(x);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }}
    ctx.stroke();

    if (hasNode) {{
      const nodePY    = toCanvasY(nodeFrac * L_len);
      ctx.strokeStyle = ORANGE;
      ctx.lineWidth   = 7;
      ctx.lineCap     = "round";
      ctx.beginPath();
      ctx.moveTo(stringsX[0] - 18, nodePY);
      ctx.lineTo(stringsX[5] + 18, nodePY);
      ctx.stroke();
    }}

    ctx.fillStyle = "#222";
    ctx.font      = "13px 'Source Sans 3', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`${{['','1st','2nd','3rd','4th','5th'][N]}} Harmonic`, W / 2, 18);
  }}

  function step(wallNow) {{
    if (lastWall === null) lastWall = wallNow;
    const wallDt = (wallNow - lastWall) / 1000.0;
    lastWall = wallNow;
    t_phys = (t_phys + wallDt * SPEED) % T_phys;
    draw();
    if (playing) rafId = requestAnimationFrame(step);
  }}

  playBtn.addEventListener("click", function() {{
    playing = !playing;
    playBtn.textContent = playing ? "Pause" : "Play";
    if (playing) {{
      lastWall = null;
      rafId = requestAnimationFrame(step);
    }}
  }});

  draw();
}})();
</script>
"""

with col_canvas:
    st.components.v1.html(canvas_html, height=600, scrolling=False)
