#!/usr/bin/env python3
"""
Generate Figure 1 (System Architecture) and Figure 5 (AOP Framework)
for the ADMET-X manuscript.

Style matches 04_generate_figures.py (same fonts, colors, DPI).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Publication style — matches other figures
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
})

# Colors
TIER1 = "#4CAF50"
TIER2 = "#2196F3"
TIER3 = "#9C27B0"
CONSENSUS = "#FF9800"
ENHANCED = "#009688"
INPUT_C = "#607D8B"
OUTPUT_C = "#FF5722"
DARK = "#455A64"
CELERY = "#E91E63"


def _box(ax, x, y, w, h, label, color, fs=8, sub=None, sub_fs=6.5,
         tc="white", ec="white", lw=1.0):
    """Draw rounded box with optional subtitle. Returns (cx, cy) center."""
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor=ec, linewidth=lw,
                           alpha=0.92, zorder=3)
    ax.add_patch(patch)
    if sub:
        ax.text(x + w / 2, y + h * 0.63, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=sub_fs, color=tc, alpha=0.9, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)
    return x + w / 2, y + h / 2


def _arr(ax, x1, y1, x2, y2, color=DARK, lw=1.5, style="->"):
    """Arrow from (x1,y1) to (x2,y2)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)


# ============================================================================
# Figure 1: System Architecture  (3 stacked panels, full width each)
# ============================================================================

def figure_1_architecture(output_dir):
    """
    (A) Three-tier prediction engine — horizontal flow
    (B) Enhanced features pipeline — 2 x 5 grid
    (C) End-to-end input/output processing flow
    """
    fig = plt.figure(figsize=(15, 18))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.22)

    # ── PANEL A: Three-Tier Prediction Engine ──────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("(A) Three-Tier Multi-Engine Prediction Architecture",
                  fontsize=13, fontweight="bold", pad=14, loc="left")

    # Light background panel
    bg = FancyBboxPatch((0.2, 0.3), 15.5, 7.2, boxstyle="round,pad=0.2",
                        facecolor="#FAFAFA", edgecolor="#E0E0E0", lw=1, zorder=0)
    ax.add_patch(bg)

    # Input molecule
    _box(ax, 0.5, 3.0, 2.0, 2.0, "Input\nMolecule", INPUT_C, fs=9,
         sub="SMILES / SDF")

    # Arrows from input to each tier
    _arr(ax, 2.5, 4.0, 3.5, 6.2, color="#90A4AE")
    _arr(ax, 2.5, 4.0, 3.5, 4.0, color="#90A4AE")
    _arr(ax, 2.5, 4.0, 3.5, 1.8, color="#90A4AE")

    # Tier boxes — stacked vertically, aligned left edges
    bw, bh = 4.0, 1.6  # box width, height

    # Tier 1
    _box(ax, 3.5, 5.6, bw, bh, "Tier 1: RDKit", TIER1, fs=10,
         sub="Rule-based | ~60 descriptors\nLogP, TPSA, Lipinski, QED", sub_fs=7)
    ax.text(7.7, 6.9, "Always available", fontsize=6.5, color=TIER1,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=TIER1, lw=0.6))

    # Tier 2
    _box(ax, 3.5, 3.2, bw, bh, "Tier 2: Deep-PK", TIER2, fs=10,
         sub="Deep learning API | 50+ endpoints\nComprehensive ADMET", sub_fs=7)
    ax.text(7.7, 4.5, "API-dependent", fontsize=6.5, color=TIER2,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=TIER2, lw=0.6))

    # Tier 3
    _box(ax, 3.5, 0.8, bw, bh, "Tier 3: Trained ML", TIER3, fs=10,
         sub="RF(500) + GB(200) ensembles\nTOX21 (AUC 0.71) & ClinTox (0.78)", sub_fs=7)
    ax.text(7.7, 2.1, "ECFP4 2048-bit", fontsize=6.5, color=TIER3,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=TIER3, lw=0.6))

    # Arrows from tiers to consensus
    _arr(ax, 7.5, 6.4, 9.5, 4.6, lw=1.8)
    _arr(ax, 7.5, 4.0, 9.5, 4.0, lw=1.8)
    _arr(ax, 7.5, 1.6, 9.5, 3.5, lw=1.8)

    # Consensus scoring box
    _box(ax, 9.5, 2.8, 3.2, 2.4, "Consensus\nScoring", CONSENSUS, fs=11,
         sub="Weighted multi-engine\nagreement scores", sub_fs=7.5)

    # Arrow to output
    _arr(ax, 12.7, 4.0, 13.5, 4.0, lw=2)

    # Output
    _box(ax, 13.5, 2.8, 2.2, 2.4, "Aggregated\nResult", OUTPUT_C, fs=9,
         sub="~250 properties\nper compound", sub_fs=7)

    # ── PANEL B: Enhanced Features Pipeline ────────────────────────
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("(B) Enhanced Translational Features Pipeline (10 Modules)",
                  fontsize=13, fontweight="bold", pad=14, loc="left")

    bg = FancyBboxPatch((0.2, 0.5), 15.5, 7.0, boxstyle="round,pad=0.2",
                        facecolor="#FAFAFA", edgecolor="#E0E0E0", lw=1, zorder=0)
    ax.add_patch(bg)

    # Source: consensus results
    _box(ax, 0.4, 5.0, 1.8, 1.4, "Consensus\nResults", CONSENSUS, fs=7.5)
    _arr(ax, 2.2, 5.7, 2.8, 5.7, color="#BDBDBD")

    # Module dimensions
    mw, mh = 2.2, 1.6   # module width, height
    gap = 0.35           # gap between modules
    step = mw + gap

    # Top row: modules 1–5
    modules_top = [
        ("1. Applicability\n   Domain", "#42A5F5", "Tanimoto distance"),
        ("2. Confidence\n   Intervals", "#66BB6A", "Conformal pred."),
        ("3. SHAP\n   Explainability", "#AB47BC", "Feature import."),
        ("4. Prediction\n   Reliability", "#EF5350", "PRI composite"),
        ("5. Clinical\n   Developability", "#FFA726", "CDS 5-dim score"),
    ]

    x0_top = 2.8
    for i, (label, color, sub) in enumerate(modules_top):
        x = x0_top + i * step
        _box(ax, x, 5.0, mw, mh, label, color, fs=7, sub=sub, sub_fs=6)
        if i < 4:
            _arr(ax, x + mw, 5.8, x + step, 5.8, color="#BDBDBD", lw=1.2)

    # Connecting arrow: row 1 → row 2 (U-turn at right, down, then left)
    last_x = x0_top + 4 * step + mw
    first_x_bot = x0_top
    mid_y_top = 5.0
    mid_y_bot = 3.2
    # Down from last top module
    _arr(ax, last_x - mw / 2, mid_y_top, last_x - mw / 2, mid_y_bot + mh,
         color="#BDBDBD", lw=1.2)

    # Bottom row: modules 6–10
    modules_bot = [
        ("6. BCS\n   Classification", "#26C6DA", "ESOL + Class I-IV"),
        ("7. Metabolite\n   Prediction", "#8D6E63", "SMARTS + DILI"),
        ("8. Drug-Drug\n   Interaction", "#EC407A", "CYP victim/perp"),
        ("9. Comparative\n   Profiling", "#78909C", "vs 23 ref drugs"),
        ("10. bRo5/PROTAC\n   & AOP Mapping", "#5C6BC0", "bRo5 + 14 AOPs"),
    ]

    # Bottom row goes right-to-left (reverse order) for clean U-turn flow
    x0_bot = x0_top
    for i, (label, color, sub) in enumerate(modules_bot):
        xi = 4 - i  # reverse: rightmost first
        x = x0_bot + xi * step
        _box(ax, x, 1.6, mw, mh, label, color, fs=7, sub=sub, sub_fs=6)

    # Arrows between bottom modules (left to right in visual, but flow is
    # right to left: 6←7←8←9←10)
    for i in range(4):
        xi_from = 4 - i       # start at rightmost (module 10)
        xi_to = 4 - (i + 1)   # go to next left
        x_from = x0_bot + xi_from * step
        x_to = x0_bot + xi_to * step + mw
        _arr(ax, x_from, 2.4, x_to, 2.4, color="#BDBDBD", lw=1.2)

    # Arrow from module 6 to output
    _arr(ax, x0_bot, 2.4, x0_bot - 0.6, 2.4, color="#BDBDBD", lw=1.2)

    # Output box (left of bottom row)
    # Actually let's put an output at the far left bottom
    # The flow is: Consensus → 1→2→3→4→5 (top L-R) → 10→9→8→7→6 (bot R-L) → Output
    # But that's confusing. Let me make both rows left-to-right instead.

    # Actually, let me redo the bottom row as left-to-right too, with the
    # connecting arrow going down from module 5, then left to module 6.

    # Clear and redo - overwrite the bottom row with left-to-right flow
    # (The FancyBboxPatches are already drawn, they're fine positionally.)
    # Actually the modules_bot with xi=4-i reversal placed them in wrong positions.

    # Let me just redraw this properly. The patches are already on the axes,
    # but I'll overlay correct ones. Easier to just accept the current layout
    # and fix the arrows.

    # Hmm, the patches were drawn at reversed positions. Let me reconsider.
    # modules_bot[0] = "6. BCS" was drawn at xi=4 (rightmost position)
    # modules_bot[4] = "10. bRo5" was drawn at xi=0 (leftmost position)
    # This means 10 is at left, 6 is at right on the bottom row.
    # Flow: ...5 → ↓ → 10(left) → 9 → 8 → 7 → 6(right) would need right arrows
    # But the arrows I drew go from right to left (from module 10's position).

    # This is getting confusing. Let me just clear and redo Panel B entirely.
    pass

    # I'll regenerate the entire figure with a cleaner approach below.

    out = Path(output_dir)
    plt.close()

    # ===================== CLEAN REGENERATION =====================
    _generate_fig1_clean(out)
    _generate_fig5_clean(out)


def _generate_fig1_clean(out):
    """Clean Figure 1 with proper alignment."""

    fig = plt.figure(figsize=(15, 18))
    gs = GridSpec(3, 1, height_ratios=[1.0, 1.0, 0.75], hspace=0.18)

    # ── PANEL A ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("(A) Three-Tier Multi-Engine Prediction Architecture",
                  fontsize=13, fontweight="bold", pad=14, loc="left")

    # Background
    ax.add_patch(FancyBboxPatch((0.1, 0.2), 15.8, 7.5,
                 boxstyle="round,pad=0.2", fc="#FAFAFA", ec="#E0E0E0", lw=1))

    # Input
    _box(ax, 0.4, 3.0, 2.0, 2.0, "Input\nMolecule", INPUT_C, fs=9,
         sub="SMILES / SDF")

    # Arrows input → tiers
    for ty in [6.4, 4.0, 1.6]:
        _arr(ax, 2.4, 4.0, 3.3, ty, color="#90A4AE", lw=1.2)

    # Tiers
    tw, th = 4.2, 1.6
    _box(ax, 3.3, 5.6, tw, th, "Tier 1: RDKit", TIER1, fs=10,
         sub="Rule-based  |  ~60 descriptors\nLogP, TPSA, Lipinski, QED", sub_fs=7)
    _box(ax, 3.3, 3.2, tw, th, "Tier 2: Deep-PK", TIER2, fs=10,
         sub="Deep learning API  |  50+ endpoints\nComprehensive ADMET", sub_fs=7)
    _box(ax, 3.3, 0.8, tw, th, "Tier 3: Trained ML", TIER3, fs=10,
         sub="RF(500) + GB(200)  |  ECFP4 2048-bit\nTOX21 (AUC 0.71) & ClinTox (0.78)", sub_fs=7)

    # Badges
    ax.text(7.7, 7.0, "Always available", fontsize=6.5, color=TIER1,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=TIER1, lw=0.6))
    ax.text(7.7, 4.6, "API-dependent", fontsize=6.5, color=TIER2,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=TIER2, lw=0.6))
    ax.text(7.7, 2.2, "Scaffold split", fontsize=6.5, color=TIER3,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=TIER3, lw=0.6))

    # Arrows tiers → consensus
    _arr(ax, 7.5, 6.4, 9.2, 4.6, lw=1.8)
    _arr(ax, 7.5, 4.0, 9.2, 4.0, lw=1.8)
    _arr(ax, 7.5, 1.6, 9.2, 3.4, lw=1.8)

    # Consensus
    _box(ax, 9.2, 2.8, 3.0, 2.4, "Consensus\nScoring", CONSENSUS, fs=11,
         sub="Weighted multi-engine\nagreement", sub_fs=7.5)

    # Arrow → output
    _arr(ax, 12.2, 4.0, 13.3, 4.0, lw=2.0)

    # Output
    _box(ax, 13.3, 2.8, 2.4, 2.4, "Aggregated\nResult", OUTPUT_C, fs=10,
         sub="~250 properties\nper compound", sub_fs=7)

    # ── PANEL B ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("(B) Enhanced Translational Features Pipeline",
                  fontsize=13, fontweight="bold", pad=14, loc="left")

    ax.add_patch(FancyBboxPatch((0.1, 0.3), 15.8, 7.3,
                 boxstyle="round,pad=0.2", fc="#FAFAFA", ec="#E0E0E0", lw=1))

    # Module dimensions
    mw, mh = 2.5, 1.5
    gx = 0.35  # horizontal gap
    step = mw + gx

    # Top row: modules 1–5  (y=5.5)
    top_y = 5.5
    top_x0 = 0.5
    top_modules = [
        ("1. Applicability\n   Domain", "#42A5F5", "Tanimoto distance"),
        ("2. Confidence\n   Intervals", "#66BB6A", "Conformal pred."),
        ("3. SHAP\n   Explainability", "#AB47BC", "Feature importance"),
        ("4. Prediction\n   Reliability", "#EF5350", "PRI (4-component)"),
        ("5. Clinical\n   Developability", "#FFA726", "CDS (5-dimension)"),
    ]
    for i, (label, color, sub) in enumerate(top_modules):
        x = top_x0 + i * step
        _box(ax, x, top_y, mw, mh, label, color, fs=7, sub=sub, sub_fs=6)
        if i < 4:
            _arr(ax, x + mw, top_y + mh / 2, x + step, top_y + mh / 2,
                 color="#9E9E9E", lw=1.2)

    # U-turn arrow: down from module 5, left to module 6
    turn_x = top_x0 + 4 * step + mw / 2
    _arr(ax, turn_x, top_y, turn_x, 4.3, color="#9E9E9E", lw=1.2)
    # Horizontal left
    bot_y = 2.5
    ax.annotate("", xy=(top_x0 + 4 * step + mw / 2, bot_y + mh),
                xytext=(turn_x, 4.3),
                arrowprops=dict(arrowstyle="-", color="#9E9E9E", lw=1.2))

    # Bottom row: modules 6–10  (y=2.5) — right to left visually
    # Module 6 at far right, module 10 at far left
    # Flow: ← 6 ← 7 ← 8 ← 9 ← 10
    bot_modules = [
        ("10. bRo5/PROTAC\n    & AOP", "#5C6BC0", "bRo5 + 14 AOPs"),
        ("9. Comparative\n   Profiling", "#78909C", "vs 23 ref drugs"),
        ("8. Drug-Drug\n   Interaction", "#EC407A", "CYP victim/perp."),
        ("7. Metabolite\n   Prediction", "#8D6E63", "SMARTS + DILI"),
        ("6. BCS\n   Classification", "#26C6DA", "ESOL, Class I–IV"),
    ]
    for i, (label, color, sub) in enumerate(bot_modules):
        x = top_x0 + i * step
        _box(ax, x, bot_y, mw, mh, label, color, fs=7, sub=sub, sub_fs=6)
        if i < 4:
            _arr(ax, x + mw, bot_y + mh / 2, x + step, bot_y + mh / 2,
                 color="#9E9E9E", lw=1.2)

    # Arrow from top row end down to bottom row start (module 10)
    _arr(ax, turn_x, bot_y + mh, top_x0 + mw / 2, bot_y + mh,
         color="#9E9E9E", lw=1.2, style="-")
    # Actual arrow into module 10
    _arr(ax, turn_x, 4.0, turn_x, bot_y + mh, color="#9E9E9E", lw=1.2,
         style="-")
    # We need a single clean arrow. Let me simplify:
    # Just draw a down-arrow from module 5 center to module 10 center level,
    # then arrow going left across to module 10.

    # Final output arrow from module 6
    last_x = top_x0 + 4 * step + mw
    _arr(ax, last_x, bot_y + mh / 2, 15.2, bot_y + mh / 2,
         color="#9E9E9E", lw=1.2)
    _box(ax, 14.8, bot_y - 0.1, 1.0, mh + 0.2, "~250\nProps", OUTPUT_C, fs=7)

    # Annotation
    ax.text(8, 1.2,
            "Each module is independently exception-handled to prevent cascade failures",
            ha="center", fontsize=8, style="italic", color="#757575")

    # Flow numbering indicator
    ax.annotate("", xy=(0.5, 4.6), xytext=(0.5, 5.5),
                arrowprops=dict(arrowstyle="<-", color=CONSENSUS, lw=1.5))
    ax.text(0.15, 4.85, "from\n(A)", fontsize=7, color=CONSENSUS,
            fontweight="bold", ha="center")

    # ── PANEL C ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(C) End-to-End Processing Flow",
                  fontsize=13, fontweight="bold", pad=14, loc="left")

    ax.add_patch(FancyBboxPatch((0.1, 0.2), 15.8, 5.5,
                 boxstyle="round,pad=0.2", fc="#FAFAFA", ec="#E0E0E0", lw=1))

    # Horizontal pipeline: Input → Validation → Celery → Engine → Features → Report
    bw, bh = 2.0, 2.4
    cy = 1.8   # center y for boxes
    positions = [0.4, 3.0, 5.6, 8.2, 10.8, 13.4]
    boxes = [
        ("User\nInput", "#78909C",
         "SMILES text\nBatch CSV\nJSME editor"),
        ("Validation\n& Parsing", "#607D8B",
         "RDKit mol check\nSMILES canon.\nDuplicate filter"),
        ("Celery\nTask Queue", CELERY,
         "Redis broker\nAsync workers\nBatch parallel"),
        ("Multi-Engine\nPrediction", CONSENSUS,
         "Tier 1 + 2 + 3\nConsensus score\n(see Panel A)"),
        ("Enhanced\nFeatures", ENHANCED,
         "10 modules\nSequential pipeline\n(see Panel B)"),
        ("Interactive\nReport", OUTPUT_C,
         "~250 properties\nWeb dashboard\nCSV export"),
    ]

    for i, (pos, (label, color, sub)) in enumerate(zip(positions, boxes)):
        _box(ax, pos, cy, bw, bh, label, color, fs=9, sub=sub, sub_fs=6.5)
        if i < len(positions) - 1:
            _arr(ax, pos + bw, cy + bh / 2, positions[i + 1], cy + bh / 2,
                 lw=2.0)

    # Time annotation
    ax.text(8, 0.8, "Processing time: ~2-5 seconds per compound (single) | "
            "~30 seconds for 100 compounds (batch via Celery)",
            ha="center", fontsize=7.5, style="italic", color="#757575")

    plt.savefig(out / "fig1_architecture.pdf")
    plt.savefig(out / "fig1_architecture.png")
    plt.close()
    print(f"  Saved fig1_architecture.pdf/png")


# ============================================================================
# Figure 5: AOP Framework
# ============================================================================

def _generate_fig5_clean(out):
    """
    (A) ADMET-to-AOP mapping overview
    (B) Example hepatotoxicity cascade (AOP:25)
    (C) AOP coverage by organ system
    """
    fig = plt.figure(figsize=(16, 12))

    # ── Panel A: ADMET-to-AOP Mapping Overview (top, full width) ──
    ax = fig.add_axes([0.03, 0.60, 0.94, 0.38])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("(A) ADMET Prediction to Adverse Outcome Pathway Mapping",
                  fontsize=12, fontweight="bold", pad=12, loc="left")

    # Left column: ADMET predictions
    ax.add_patch(FancyBboxPatch((0.2, 1.3), 3.8, 6.2,
                 boxstyle="round,pad=0.15", fc="#ECEFF1", ec="#CFD8DC", lw=0.8))
    ax.text(2.1, 7.2, "ADMET Predictions", fontsize=10, fontweight="bold",
            ha="center", color=DARK)

    preds = [
        ("Toxicity predictions", "#EF5350"),
        ("CYP inhibition", "#EC407A"),
        ("Reactive metabolites", "#8D6E63"),
        ("Structural alerts", "#FF7043"),
        ("TOX21 endpoints", "#AB47BC"),
        ("Physicochemical (LogP)", "#42A5F5"),
    ]
    for i, (label, color) in enumerate(preds):
        y = 6.5 - i * 0.85
        _box(ax, 0.4, y - 0.25, 3.4, 0.6, label, color, fs=7.5)

    # Center: Mapping engine
    _box(ax, 5.5, 2.8, 3.5, 2.8, "AOP Mapping\nEngine", DARK, fs=11,
         sub="14 curated AOPs\nStructural alerts + triggers\nConfidence scoring",
         sub_fs=7)

    # Arrows left → center
    for i in range(6):
        y = 6.5 - i * 0.85
        _arr(ax, 3.8, y, 5.5, 4.2, color="#BDBDBD", lw=1)

    # AOP-Wiki reference
    _box(ax, 5.8, 6.2, 2.8, 1.0, "AOP-Wiki\nKnowledge Base", "#78909C",
         fs=8, sub="OECD endorsed", sub_fs=6.5)
    _arr(ax, 7.2, 6.2, 7.2, 5.6, color="#90A4AE")

    # Arrow center → right
    _arr(ax, 9.0, 4.2, 10.5, 4.2, lw=2)

    # Right: Organ-specific AOPs
    ax.add_patch(FancyBboxPatch((10.3, 0.6), 9.2, 7.0,
                 boxstyle="round,pad=0.15", fc="#FFF8E1", ec="#FFE082", lw=0.8))
    ax.text(14.9, 7.2, "Organ-Specific Adverse Outcome Pathways",
            fontsize=10, fontweight="bold", ha="center", color=DARK)

    organs = [
        ("Hepatotoxicity", ["AOP:18 Protein alkylation \u2192 Fibrosis",
                            "AOP:25 Mito. dysfunction \u2192 DILI",
                            "AOP:34 BSEP inhibition \u2192 Cholestasis"],
         "#EF5350", 6.2),
        ("Cardiotoxicity", ["AOP:150 hERG inhibition \u2192 Arrhythmia",
                            "AOP:371 Ion channel block \u2192 QT prolong."],
         "#E91E63", 4.5),
        ("Nephrotoxicity", ["AOP:263 Transporter inh. \u2192 AKI",
                            "ROS generation \u2192 Glomerular injury",
                            "Crystal precipitation \u2192 AKI"],
         "#FF9800", 2.8),
        ("Genotox / Endocrine\n/ Skin Sensit.", [
            "AOP:15 DNA alkylation \u2192 Cancer",
            "AOP:19/29 ER/AR binding \u2192 Repro. tox",
            "AOP:40 Haptenation \u2192 Sensitization"],
         "#AB47BC", 1.0),
    ]
    for organ, aops, color, y in organs:
        _box(ax, 10.5, y, 2.8, 1.3, organ, color, fs=7.5)
        for j, txt in enumerate(aops):
            ax.text(13.6, y + 1.0 - j * 0.38, txt, fontsize=6.5,
                    color=DARK, va="center")

    # ── Panel B: Example AOP Cascade (bottom left) ─────────────────
    ax = fig.add_axes([0.03, 0.05, 0.57, 0.48])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(B) Example: Mitochondrial Dysfunction \u2192 DILI (AOP:25)",
                  fontsize=11, fontweight="bold", pad=10, loc="left")

    # Stage labels
    stages = [
        ("MIE", "#FFF9C4"), ("Key Event 1", "#FFE0B2"),
        ("Key Event 2", "#FFCCBC"), ("Key Event 3", "#FFCDD2"),
        ("Adverse Outcome", "#EF9A9A"),
    ]
    events = [
        ("Inhibition of\nmitochondrial\ncomplex I/III", "#FFF9C4", "#795548"),
        ("Decreased ATP\nproduction", "#FFE0B2", "#E65100"),
        ("Increased ROS +\nmembrane\ndamage", "#FFCCBC", "#BF360C"),
        ("Hepatocyte\napoptosis /\nnecrosis", "#FFCDD2", "#C62828"),
        ("Drug-Induced\nLiver Injury\n(DILI)", "#EF9A9A", "#B71C1C"),
    ]

    ew, eh = 2.4, 2.2
    ex0 = 0.5
    estep = 3.0
    ey = 6.5

    for i, ((stg, stg_c), (txt, bg_c, tc)) in enumerate(zip(stages, events)):
        x = ex0 + i * estep
        # Stage label
        ax.text(x + ew / 2, ey + eh + 0.4, stg, fontsize=7.5,
                fontweight="bold", ha="center", color=DARK,
                bbox=dict(boxstyle="round,pad=0.15", fc=stg_c, ec="#BDBDBD"))
        # Event box
        patch = FancyBboxPatch((x, ey), ew, eh, boxstyle="round,pad=0.12",
                               fc=bg_c, ec=DARK, lw=1.2, zorder=3)
        ax.add_patch(patch)
        ax.text(x + ew / 2, ey + eh / 2, txt, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=tc, zorder=4)
        # Arrow to next
        if i < 4:
            _arr(ax, x + ew, ey + eh / 2, x + estep, ey + eh / 2,
                 color=DARK, lw=2)

    # Evidence layer
    ax.text(7.5, 5.3, "Triggering Evidence from ADMET-X",
            fontsize=9, fontweight="bold", ha="center", color=DARK)

    evidence = [
        (ex0 + 0 * estep + ew / 2, "LogP > 3\n(mito. accumulation)", "#42A5F5"),
        (ex0 + 1 * estep + ew / 2, "TOX21: SR-MMP\n(mito. membrane)", "#AB47BC"),
        (ex0 + 2 * estep + ew / 2, "TOX21: SR-ARE\n(oxidative stress)", "#66BB6A"),
        (ex0 + 3 * estep + ew / 2, "Hepatotoxicity ML\nmodel prediction", "#EF5350"),
        (ex0 + 4 * estep + ew / 2, "Clinical DILI\nclass association", "#FF7043"),
    ]
    for x, txt, color in evidence:
        _arr(ax, x, 4.8, x, ey, color=color, lw=1, style="-|>")
        ax.text(x, 3.8, txt, ha="center", va="center", fontsize=6.5,
                color=DARK,
                bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="white",
                          alpha=0.15))

    # Confidence bar
    cx, cy = 0.5, 1.5
    ax.text(cx, cy + 0.8, "AOP Confidence:", fontsize=8, fontweight="bold",
            color=DARK)
    bar_len = 12
    for i in range(100):
        f = i / 100
        ax.barh(cy, bar_len / 100, left=cx + f * bar_len, height=0.35,
                color=(0.4 + 0.6 * f, 0.8 - 0.6 * f, 0.2), ec="none")
    ax.text(cx, cy - 0.35, "0 (Low)", fontsize=7, color="#4CAF50")
    ax.text(cx + bar_len, cy - 0.35, "1 (High)", fontsize=7, color="#F44336",
            ha="right")
    marker = 0.68
    mx = cx + marker * bar_len
    ax.plot(mx, cy + 0.17, "v", ms=10, color=DARK)
    ax.text(mx, cy + 0.75, f"Score = {marker} (Moderate)", fontsize=8,
            fontweight="bold", ha="center", color="#D84315")

    # ── Panel C: AOP Coverage (bottom right) ───────────────────────
    ax = fig.add_axes([0.63, 0.05, 0.34, 0.48])
    ax.set_title("(C) AOP Coverage by Organ System",
                  fontsize=11, fontweight="bold", pad=10, loc="left")

    data = [
        ("Liver", 3, 0), ("Heart", 1, 1), ("Kidney", 1, 2),
        ("Systemic\n(Genotox)", 2, 0), ("Endocrine", 2, 0),
        ("Skin", 1, 0), ("Neuro", 0, 1),
    ]
    names = [d[0] for d in data]
    endorsed = [d[1] for d in data]
    research = [d[2] for d in data]
    yp = np.arange(len(names))

    ax.barh(yp, endorsed, height=0.55, color="#4CAF50", ec="white",
            label="OECD Endorsed")
    ax.barh(yp, research, height=0.55, left=endorsed, color="#FFC107",
            ec="white", label="Under Review")

    for i, (e, r) in enumerate(zip(endorsed, research)):
        ax.text(e + r + 0.1, i, str(e + r), va="center", fontsize=9,
                fontweight="bold")

    ax.set_yticks(yp)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Number of AOPs")
    ax.set_xlim(0, 5)
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()

    ax.text(3.2, 7.3, "Total: 14 AOPs\n6 organs\n10 OECD endorsed",
            fontsize=9, ha="center", va="center", color=DARK,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#4CAF50"))

    plt.savefig(out / "fig5_aop.pdf")
    plt.savefig(out / "fig5_aop.png")
    plt.close()
    print(f"  Saved fig5_aop.pdf/png")


# ============================================================================
if __name__ == "__main__":
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Generating Figure 1: System Architecture...")
    print("Generating Figure 5: AOP Framework...")
    figure_1_architecture(str(output_dir))
    print("\nDone! Both figures saved to:", output_dir)
