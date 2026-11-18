"""
Sentinel-Cap | Publication-Grade Figure 2 Generator
Purged and Embargoed Rolling Walk-Forward Validation Diagram (300 DPI)
Sized for 0.25-page placement in Section 5.1 (Experimental Setup).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_figure2() -> Path:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    # 12.0 x 3.85 in aspect ratio -> occupies ~0.25 pages at full page width in Word
    fig = plt.figure(figsize=(12.0, 3.85), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])

    # =========================================================================
    # LEFT PANEL: 25-YEAR CHRONOLOGICAL EXPANDING WALK-FORWARD FOLDS (2000–2025)
    # =========================================================================
    ax_main.set_facecolor("#ffffff")
    ax_main.set_xlim(1999.6, 2026.3)
    ax_main.set_ylim(0.15, 5.25)

    # Era background shading
    ax_main.axvspan(2000.0, 2008.0, color="#f1f5f9", alpha=0.85, zorder=0)
    ax_main.axvspan(2008.0, 2025.95, color="#f8fafc", alpha=0.55, zorder=0)
    ax_main.axvline(2008.0, color="#334155", linestyle="--", linewidth=1.1, zorder=2)

    # Top Era Headers
    ax_main.text(
        2004.0,
        4.88,
        "8-Year Warm-Up\n(2000–2007)",
        ha="center",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#e2e8f0", edgecolor="#94a3b8", lw=0.8),
    )
    ax_main.text(
        2017.0,
        4.88,
        "Out-of-Sample Walk-Forward Evaluation (2008–2025 | Quarterly Step $\\Delta T = 63$ Trading Days)",
        ha="center",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color="#0f766e",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#f0fdfa", edgecolor="#5eead4", lw=0.8),
    )

    # Representative Walk-Forward Folds: (y_pos, label, train_end, oos_end, note)
    folds = [
        (3.85, "Fold 1\n(2008 Q1)", 2007.80, 2009.10, "GFC Shock"),
        (2.95, "Fold 2\n(2008 Q2)", 2008.05, 2009.35, "+63d Step"),
        (2.05, "Fold $k$\n(2020 Q1)", 2019.80, 2021.10, "COVID-19"),
        (1.15, "Fold $K$\n(2025 Q1)", 2024.75, 2025.95, "Final Fold"),
    ]

    bar_h = 0.56
    purge_w = 0.22   # Visual width for h-day purge zone
    embargo_w = 0.22 # Visual width for h_emb = h + 5 embargo buffer

    for y, label, tr_end, oos_end, note in folds:
        # 1. Expanding Training Window
        ax_main.add_patch(
            Rectangle((2000.0, y - bar_h / 2), tr_end - 2000.0, bar_h, facecolor="#2563eb", edgecolor="#1e40af", lw=0.8, zorder=3)
        )
        ax_main.text(
            2000.0 + (tr_end - 2000.0) / 2.0,
            y,
            "Expanding Training Window $\\mathcal{T}_{\\mathrm{train}}$",
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold",
            color="#ffffff",
            zorder=4,
        )

        # 2. Purged Label-Overlap Zone
        ax_main.add_patch(
            Rectangle((tr_end, y - bar_h / 2), purge_w, bar_h, facecolor="#dc2626", edgecolor="#991b1b", lw=0.7, zorder=4)
        )

        # 3. Embargo Buffer
        t_test = tr_end + purge_w
        ax_main.add_patch(
            Rectangle((t_test, y - bar_h / 2), embargo_w, bar_h, facecolor="#f59e0b", edgecolor="#b45309", lw=0.7, zorder=4)
        )

        # 4. Out-of-Sample Test Fold
        oos_start = t_test + embargo_w
        ax_main.add_patch(
            Rectangle((oos_start, y - bar_h / 2), oos_end - oos_start, bar_h, facecolor="#0d9488", edgecolor="#115e59", lw=0.8, zorder=3)
        )
        ax_main.text(
            oos_start + (oos_end - oos_start) / 2.0,
            y,
            "OOS",
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="bold",
            color="#ffffff",
            zorder=4,
        )

        # 5. Masked Future Timeline
        if oos_end < 2025.95:
            ax_main.add_patch(
                Rectangle(
                    (oos_end, y - bar_h / 2),
                    2025.95 - oos_end,
                    bar_h,
                    facecolor="#e2e8f0",
                    edgecolor="#cbd5e1",
                    lw=0.6,
                    linestyle="--",
                    zorder=2,
                )
            )
            ax_main.text(
                oos_end + 0.25,
                y,
                note,
                ha="left",
                va="center",
                fontsize=6.3,
                color="#475569",
                style="italic",
                zorder=4,
            )

    # Ellipsis indicator between Fold 2 and Fold k
    ax_main.text(2012.5, 2.50, "⋮   Quarterly Expanding Splits ($\\Delta T = 63$ d)   ⋮", ha="center", va="center", fontsize=7.0, color="#475569", fontweight="bold")

    ax_main.set_yticks([f[0] for f in folds])
    ax_main.set_yticklabels([f[1] for f in folds], fontsize=7.2, fontweight="bold", color="#1e293b")
    ax_main.set_xticks([2000, 2004, 2008, 2012, 2016, 2020, 2025])
    ax_main.set_xticklabels(["2000", "2004", "2008", "2012", "2016", "2020", "2025"], fontsize=7.3, color="#1e293b")
    ax_main.set_xlabel("Chronological Timeline (~6,450 NSE Trading Days)", fontsize=7.6, fontweight="bold", color="#0f172a")
    ax_main.set_title("(a) Purged & Embargoed Walk-Forward Schedule (2000–2025)", fontsize=8.6, fontweight="bold", color="#0f172a", pad=6)

    for spine in ["top", "right"]:
        ax_main.spines[spine].set_visible(False)
    ax_main.spines["left"].set_color("#94a3b8")
    ax_main.spines["bottom"].set_color("#94a3b8")

    # Bottom Legend inside main axis
    legend_elements = [
        Patch(facecolor="#2563eb", edgecolor="#1e40af", label="Train ($\\mathcal{T}_{\\mathrm{train}}$)"),
        Patch(facecolor="#dc2626", edgecolor="#991b1b", label="Purge ($h$ d)"),
        Patch(facecolor="#f59e0b", edgecolor="#b45309", label="Embargo ($h+5$ d)"),
        Patch(facecolor="#0d9488", edgecolor="#115e59", label="OOS Test"),
        Patch(facecolor="#e2e8f0", edgecolor="#cbd5e1", label="Masked Future"),
    ]
    ax_main.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fontsize=6.3,
        handlelength=1.1,
        columnspacing=0.9,
    )

    # =========================================================================
    # RIGHT PANEL: MAGNIFIED SPLIT-BOUNDARY LEAKAGE PREVENTION DETAIL
    # =========================================================================
    ax_zoom.set_facecolor("#f8fafc")
    ax_zoom.set_xlim(0, 10)
    ax_zoom.set_ylim(0, 10)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor("#94a3b8")
        spine.set_linewidth(1.0)

    ax_zoom.set_title("(b) Split Boundary Detail ($h \\in \\{1, 5, 20\\}$ d)", fontsize=8.6, fontweight="bold", color="#0f172a", pad=6)

    # Top comparison: Naive Split (Leakage)
    ax_zoom.text(0.4, 8.95, "Naive Split (Label Overlap Leakage):", fontsize=7.0, fontweight="bold", color="#991b1b")
    ax_zoom.add_patch(Rectangle((0.4, 7.65), 4.6, 0.85, facecolor="#93c5fd", edgecolor="#2563eb", lw=0.8))
    ax_zoom.text(2.7, 8.07, "Train up to $t_{\\mathrm{test}}-1$", ha="center", va="center", fontsize=6.4, color="#1e3a8a", fontweight="bold")
    ax_zoom.add_patch(Rectangle((5.0, 7.65), 4.6, 0.85, facecolor="#99f6e4", edgecolor="#0d9488", lw=0.8))
    ax_zoom.text(7.7, 8.07, "OOS Test Fold", ha="center", va="center", fontsize=6.4, color="#115e59", fontweight="bold")
    # Overlap warning bar
    ax_zoom.add_patch(
        FancyBboxPatch((4.5, 7.52), 1.8, 1.1, boxstyle="round,pad=0.05", facecolor="#fecaca", edgecolor="#dc2626", lw=1.2, linestyle="--", alpha=0.85)
    )
    ax_zoom.text(5.4, 7.05, "$h$-day target $y_{s,t+h}$ leaks into test!", ha="center", va="center", fontsize=6.2, color="#b91c1c", fontweight="bold")

    # Bottom comparison: Sentinel-Cap PE-WFCV
    ax_zoom.text(0.4, 5.85, "Sentinel-Cap PE-WFCV (Zero Leakage):", fontsize=7.0, fontweight="bold", color="#0f766e")
    ax_zoom.add_patch(Rectangle((0.4, 4.55), 3.3, 0.85, facecolor="#2563eb", edgecolor="#1e40af", lw=0.8))
    ax_zoom.text(2.05, 4.97, "Valid $\\mathcal{T}_{\\mathrm{train}}$", ha="center", va="center", fontsize=6.5, color="#ffffff", fontweight="bold")

    ax_zoom.add_patch(Rectangle((3.7, 4.55), 1.3, 0.85, facecolor="#dc2626", edgecolor="#991b1b", lw=0.8))
    ax_zoom.text(4.35, 4.97, "Purge\n($h$ d)", ha="center", va="center", fontsize=6.0, color="#ffffff", fontweight="bold")

    ax_zoom.add_patch(Rectangle((5.0, 4.55), 1.5, 0.85, facecolor="#f59e0b", edgecolor="#b45309", lw=0.8))
    ax_zoom.text(5.75, 4.97, "Embargo\n($h+5$ d)", ha="center", va="center", fontsize=5.9, color="#ffffff", fontweight="bold")

    ax_zoom.add_patch(Rectangle((6.5, 4.55), 3.1, 0.85, facecolor="#0d9488", edgecolor="#115e59", lw=0.8))
    ax_zoom.text(8.05, 4.97, "Clean OOS", ha="center", va="center", fontsize=6.5, color="#ffffff", fontweight="bold")

    # Split marker line
    ax_zoom.plot([5.0, 5.0], [4.25, 5.65], color="#0f172a", linestyle=":", linewidth=1.2)
    ax_zoom.text(5.0, 4.02, "$t_{\\mathrm{test}}$", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#0f172a")

    # Formula & In-Fold Governance Box
    ax_zoom.text(
        5.0,
        2.15,
        "$\\mathcal{T}_{\\mathrm{train}} \\cap [t_{\\mathrm{test}}-h,\\; t_{\\mathrm{test}}+h_{\\mathrm{emb}}] = \\emptyset$\n"
        "• Scalers, GARCH, HMM & rolling betas\n"
        "  fit strictly inside $\\mathcal{T}_{\\mathrm{train}}$ per fold.",
        ha="center",
        va="center",
        fontsize=6.3,
        color="#1e293b",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e1", lw=0.9),
    )

    out_dir = PROJECT_ROOT / "paper_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure2_purged_walkforward.png"
    plt.subplots_adjust(left=0.065, right=0.99, top=0.89, bottom=0.14)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved 300-DPI Figure 2 to {out_path}")
    return out_path


if __name__ == "__main__":
    generate_figure2()