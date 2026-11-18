"""
Sentinel-Cap | Publication-Grade Figure 1 Generator
Three-Stage Hybrid R-Python System Architecture Flowchart (300 DPI)
Sized for 0.45-page placement in Section 4 (Proposed Methodology).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def draw_box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str,
    edgecolor: str,
    lw: float = 1.4,
    rounding: float = 0.6,
    zorder: int = 2,
    linestyle: str = "-",
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.15,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#334155",
    lw: float = 1.6,
    rad: float = 0.0,
    style: str = "-|>",
    zorder: int = 5,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=13,
        color=color,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def generate_figure1() -> Path:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    # Aspect ratio 12.0 x 5.45 -> occupies ~0.45 pages when scaled to full page width in Word
    fig, ax = plt.subplots(figsize=(12.0, 5.45), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    # Background canvas
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # -------------------------------------------------------------------------
    # Top Banner Title
    # -------------------------------------------------------------------------
    ax.text(
        50.0,
        44.3,
        "SENTINEL-CAP: THREE-STAGE HYBRID R–PYTHON QUANTITATIVE ARCHITECTURE",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        50.0,
        42.4,
        "Look-Ahead-Free Geopolitical Shock Transmission, Regime Conditioning & Constrained Black-Litterman Sector Allocation (2000–2025)",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#475569",
    )

    # -------------------------------------------------------------------------
    # Column Layout Coordinates (x, width)
    # -------------------------------------------------------------------------
    # Col 0: Raw Multi-Modal Data Streams (x=1.2, w=18.2)
    # Col 1: Stage 1 - R Statistical Refinery (x=22.6, w=22.4)
    # Col 2: Stage 2 - Python NLP & Regimes (x=48.2, w=23.6)
    # Col 3: Stage 3 - Python Hybrid ML & BL Allocator (x=75.0, w=23.8)

    # =========================================================================
    # COLUMN 0: MULTI-MODAL INPUT STREAMS
    # =========================================================================
    draw_box(ax, 1.2, 5.8, 18.2, 34.4, "#f8fafc", "#94a3b8", lw=1.2, rounding=0.8, zorder=1)
    draw_box(ax, 1.2, 37.3, 18.2, 2.9, "#334155", "#334155", lw=1.0, rounding=0.5, zorder=2)
    ax.text(
        10.3,
        38.75,
        "DATA INGESTION LAYER",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#ffffff",
        zorder=3,
    )

    # Input Card A: Market & Macro
    draw_box(ax, 2.2, 22.8, 16.2, 13.0, "#ffffff", "#64748b", lw=1.1, rounding=0.5, zorder=2)
    ax.text(10.3, 34.4, "NSE Sector & Macro Panel", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#0f172a")
    ax.text(
        10.3,
        28.4,
        "• 8 NSE Indices / Safe Havens\n"
        "  (Bank, IT, Energy, Infra,\n"
        "   FMCG, Pharma, Gold, Gilt)\n"
        "• Log Excess Returns $y_{s,t+h}$\n"
        "  for $h \\in \\{1, 5, 20\\}$ (Eqs. 1–3)\n"
        "• Global Controls: VIX, Brent\n"
        "  Crude, USD/INR, US 10Y",
        ha="center",
        va="center",
        fontsize=6.6,
        color="#1e293b",
        linespacing=1.28,
    )

    # Input Card B: Unstructured Geo & News
    draw_box(ax, 2.2, 7.2, 16.2, 14.0, "#ffffff", "#64748b", lw=1.1, rounding=0.5, zorder=2)
    ax.text(10.3, 19.8, "Geopolitical & News Feeds", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#0f172a")
    ax.text(
        10.3,
        13.2,
        "• GDELT 1.0/2.0 CAMEO Events\n"
        "  across $G=5$ Global Nodes\n"
        "  (US, EU, ME, China, India)\n"
        "• Caldara & Iacoviello Daily\n"
        "  GPR (Acts & Threats)\n"
        "• 24/7 Timestamped Financial\n"
        "  & Macro News Headlines",
        ha="center",
        va="center",
        fontsize=6.6,
        color="#1e293b",
        linespacing=1.28,
    )

    # =========================================================================
    # COLUMN 1: STAGE 1 - CLASSICAL ECONOMETRIC BASELINES (R LAYER)
    # =========================================================================
    draw_box(ax, 22.6, 5.8, 22.4, 34.4, "#eff6ff", "#1d4ed8", lw=1.5, rounding=0.8, zorder=1)
    draw_box(ax, 22.6, 37.3, 22.4, 2.9, "#1e40af", "#1e40af", lw=1.0, rounding=0.5, zorder=2)
    ax.text(
        33.8,
        38.75,
        "STAGE 1: R STATISTICAL REFINERY",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#ffffff",
        zorder=3,
    )

    # Stage 1 Sub-box 1: Chronological Alignment
    draw_box(ax, 23.7, 26.8, 20.2, 9.2, "#ffffff", "#3b82f6", lw=1.1, rounding=0.5, zorder=2)
    ax.text(33.8, 34.7, "1A. Look-Ahead-Free Alignment", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#1e3a8a")
    ax.text(
        33.8,
        30.5,
        "• 15:30 IST Exchange Cutoff Shift ($t+1$)\n"
        "• Weekend / Holiday Accumulation &\n"
        "  zoo::na.locf Pre-Market Fill (Eq. 5)\n"
        "• Sparsity (<1.5%) & Collinearity Pruning",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#1e293b",
        linespacing=1.28,
    )

    # Stage 1 Sub-box 2: Unit Root & Shrinkage
    draw_box(ax, 23.7, 17.5, 20.2, 8.0, "#ffffff", "#3b82f6", lw=1.1, rounding=0.5, zorder=2)
    ax.text(33.8, 24.3, "1B. Stationarity & Regularization", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#1e3a8a")
    ax.text(
        33.8,
        20.6,
        "• ADF ($p<0.01$) & KPSS Unit-Root Tests\n"
        "• Elastic Net ($L_1/L_2$) Shrinkage (Eq. 6)\n"
        "• Strict In-Fold Rolling Calibration",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#1e293b",
        linespacing=1.28,
    )

    # Stage 1 Sub-box 3: Econometric Priors
    draw_box(ax, 23.7, 7.2, 20.2, 9.0, "#dbeafe", "#2563eb", lw=1.2, rounding=0.5, zorder=2)
    ax.text(33.8, 14.9, "1C. Econometric Priors (fable/vars/rugarch)", ha="center", va="center", fontsize=7.3, fontweight="bold", color="#1e3a8a")
    ax.text(
        33.8,
        10.7,
        "• Univariate ARMA($p,q$) & Multivariate VAR($p$)\n"
        "• Conditional Volatility GARCH(1,1) (Eqs. 7–9)\n"
        "• Exports Priors ($\\hat{r}_{s,t+h}^{\\mathrm{R}}, \\hat{\\sigma}_{s,t+h}^{\\mathrm{R}}$) via Parquet",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#0f172a",
        linespacing=1.28,
    )

    # =========================================================================
    # COLUMN 2: STAGE 2 - NLP SPILLOVER & REGIME DISCOVERY (PYTHON LAYER)
    # =========================================================================
    draw_box(ax, 48.2, 5.8, 23.6, 34.4, "#f0fdfa", "#0f766e", lw=1.5, rounding=0.8, zorder=1)
    draw_box(ax, 48.2, 37.3, 23.6, 2.9, "#0f766e", "#0f766e", lw=1.0, rounding=0.5, zorder=2)
    ax.text(
        60.0,
        38.75,
        "STAGE 2: NLP & REGIME DISCOVERY (PYTHON)",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color="#ffffff",
        zorder=3,
    )

    # Stage 2 Sub-box 1: Transformer NLP
    draw_box(ax, 49.3, 26.8, 21.4, 9.2, "#ffffff", "#14b8a6", lw=1.1, rounding=0.5, zorder=2)
    ax.text(60.0, 34.7, "2A. Multi-Class Geopolitical NLP", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#115e59")
    ax.text(
        60.0,
        30.5,
        "• Zero-Shot bart-large-mnli ($C=6$ Shocks)\n"
        "• ProsusAI/finbert Net Polarity $S_{i,t}$ &\n"
        "  Shannon Entropy Uncertainty $U_{i,t}$ (Eq. 10)\n"
        "• Goldstein Severity $|I_i|$ & Mention Volume $q_i$",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#1e293b",
        linespacing=1.28,
    )

    # Stage 2 Sub-box 2: Spillover Decay & Betas
    draw_box(ax, 49.3, 17.5, 21.4, 8.0, "#ffffff", "#14b8a6", lw=1.1, rounding=0.5, zorder=2)
    ax.text(60.0, 24.3, "2B. Spillover Kernel & Sector Betas", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#115e59")
    ax.text(
        60.0,
        20.6,
        "• 5-Node Half-Life Decay Kernel $E_{s,c,t}^{\\mathrm{spill}}$ (Eq. 4)\n"
        "• 252-Day Rolling Event-Study Regressions\n"
        "  $\\rightarrow$ Dynamic Sector Betas $\\beta_{s,c,t}$ (Eq. 11)",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#1e293b",
        linespacing=1.28,
    )

    # Stage 2 Sub-box 3: Unsupervised Regimes
    draw_box(ax, 49.3, 7.2, 21.4, 9.0, "#ccfbf1", "#0d9488", lw=1.2, rounding=0.5, zorder=2)
    ax.text(60.0, 14.9, "2C. Unsupervised Regime Filter (GMM/HMM)", ha="center", va="center", fontsize=7.3, fontweight="bold", color="#115e59")
    ax.text(
        60.0,
        10.7,
        "• GMM-Initialized 3-State HMM on Macro $\\mathbf{m}_t$\n"
        "• Causal Forward Filter (No Future Smoothing)\n"
        "• Posterior State Vector $\\xi_t$ (Calm/Stress/Crisis)",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#0f172a",
        linespacing=1.28,
    )

    # =========================================================================
    # COLUMN 3: STAGE 3 - HYBRID ML & BLACK-LITTERMAN ALLOCATOR (PYTHON LAYER)
    # =========================================================================
    draw_box(ax, 75.0, 5.8, 23.8, 34.4, "#fffbeb", "#b45309", lw=1.5, rounding=0.8, zorder=1)
    draw_box(ax, 75.0, 37.3, 23.8, 2.9, "#b45309", "#b45309", lw=1.0, rounding=0.5, zorder=2)
    ax.text(
        86.9,
        38.75,
        "STAGE 3: HYBRID ML & BL ALLOCATOR",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color="#ffffff",
        zorder=3,
    )

    # Stage 3 Sub-box 1: Multimodal Gradient Boosting
    draw_box(ax, 76.1, 26.8, 21.6, 9.2, "#ffffff", "#d97706", lw=1.1, rounding=0.5, zorder=2)
    ax.text(86.9, 34.7, "3A. Regime-Conditioned Boosting", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#92400e")
    ax.text(
        86.9,
        30.5,
        "• Multimodal State $\\mathbf{X}_{s,t} = [E^{\\mathrm{spill}}, S, U, \\beta, \\xi, \\hat{r}^{\\mathrm{R}}, \\hat{\\sigma}^{\\mathrm{R}}, \\mathbf{m}]$\n"
        "• Regularized XGBoost / LightGBM (Eq. 13)\n"
        "  (Benchmarked vs. PyTorch LSTM)",
        ha="center",
        va="center",
        fontsize=6.4,
        color="#1e293b",
        linespacing=1.28,
    )

    # Stage 3 Sub-box 2: R-Python Hybrid & TreeSHAP
    draw_box(ax, 76.1, 17.5, 21.6, 8.0, "#ffffff", "#d97706", lw=1.1, rounding=0.5, zorder=2)
    ax.text(86.9, 24.3, "3B. Hybrid Blend & Clustered TreeSHAP", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#92400e")
    ax.text(
        86.9,
        20.6,
        "• Dynamic Ensemble $\\hat{\\mu}_{s,t+h}$ (Eq. 14)\n"
        "• Hierarchical Clustered SHAP $\\Phi_{C_k}$ (Eq. 15)\n"
        "  (Eliminates Collinear Shock Dilution)",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#1e293b",
        linespacing=1.28,
    )

    # Stage 3 Sub-box 3: Black-Litterman Optimizer
    draw_box(ax, 76.1, 7.2, 21.6, 9.0, "#fef3c7", "#b45309", lw=1.3, rounding=0.5, zorder=2)
    ax.text(86.9, 14.9, "3C. Black-Litterman Portfolio Optimizer", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#78350f")
    ax.text(
        86.9,
        10.7,
        "• Views $Q_t = \\hat{\\mu}_{t+h}$; Uncertainty $\\Omega_t \\propto U_t$\n"
        "• Regime Risk Aversion $\\lambda(Z_t)$ & TC Penalty $\\kappa$\n"
        "• Output: Optimal ETF Weights $\\mathbf{w}_t^*$ (Eq. 16)",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#0f172a",
        linespacing=1.28,
    )

    # =========================================================================
    # INTER-STAGE & INTRA-STAGE CONNECTING ARROWS
    # =========================================================================
    # Inputs -> Stage 1
    draw_arrow(ax, (18.6, 30.8), (23.5, 31.4), color="#334155", lw=1.6)
    draw_arrow(ax, (18.6, 15.8), (23.5, 29.8), color="#334155", lw=1.4, rad=-0.12)

    # Stage 1 internal vertical flow
    draw_arrow(ax, (33.8, 26.6), (33.8, 25.7), color="#1d4ed8", lw=1.5)
    draw_arrow(ax, (33.8, 17.3), (33.8, 16.4), color="#1d4ed8", lw=1.5)

    # Stage 1 -> Stage 2 (Parquet tensor handoff)
    draw_arrow(ax, (44.1, 31.4), (49.1, 31.4), color="#0f766e", lw=1.6)
    draw_arrow(ax, (44.1, 21.5), (49.1, 21.5), color="#0f766e", lw=1.6)
    draw_arrow(ax, (44.1, 11.7), (49.1, 11.7), color="#0f766e", lw=1.6)

    # Stage 2 internal vertical flow
    draw_arrow(ax, (60.0, 26.6), (60.0, 25.7), color="#0f766e", lw=1.5)

    # Stage 2 -> Stage 3 (Features, Betas & Regimes into XGBoost)
    draw_arrow(ax, (70.9, 31.4), (75.9, 32.2), color="#b45309", lw=1.6)
    draw_arrow(ax, (70.9, 21.5), (75.9, 30.4), color="#b45309", lw=1.5, rad=-0.10)
    draw_arrow(ax, (70.9, 11.7), (75.9, 28.6), color="#b45309", lw=1.5, rad=-0.18)

    # Stage 1 Priors -> Stage 3 Hybrid Blend (Bottom bypass arc)
    draw_arrow(ax, (44.1, 9.2), (75.9, 19.8), color="#1d4ed8", lw=1.3, rad=0.14)
    ax.text(
        59.8,
        5.0,
        "R Econometric Priors ($\\hat{r}_{s,t+h}^{\\mathrm{R}}, \\hat{\\sigma}_{s,t+h}^{\\mathrm{R}}$) via Apache Parquet",
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color="#1e40af",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#eff6ff", edgecolor="#93c5fd", lw=0.8),
        zorder=6,
    )

    # Stage 3 internal vertical flow
    draw_arrow(ax, (86.9, 26.6), (86.9, 25.7), color="#b45309", lw=1.5)
    draw_arrow(ax, (86.9, 17.3), (86.9, 16.4), color="#b45309", lw=1.5)

    # =========================================================================
    # BOTTOM GOVERNANCE & DEPLOYMENT BAR
    # =========================================================================
    draw_box(ax, 1.2, 0.7, 97.6, 2.8, "#f1f5f9", "#64748b", lw=1.1, rounding=0.5, linestyle="--", zorder=1)
    ax.text(
        50.0,
        2.1,
        "VALIDATION & PRODUCTION GOVERNANCE:   Purged & Embargoed Rolling Walk-Forward CV (2008–2025, $h_{\\mathrm{emb}}=h+5$ days)   |   "
        "15 bps Transaction Costs   |   Dockerized R + FastAPI + React + Gemini RAG Terminal",
        ha="center",
        va="center",
        fontsize=6.9,
        fontweight="bold",
        color="#1e293b",
        zorder=3,
    )

    out_dir = PROJECT_ROOT / "paper_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure1_architecture_flowchart.png"
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved 300-DPI Figure 1 to {out_path}")
    return out_path


if __name__ == "__main__":
    generate_figure1()