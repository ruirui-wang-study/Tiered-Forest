import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------- Helper function ----------
def draw_box(x, y, w, h, text, fc, ec="#888888"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1,
        edgecolor=ec,
        facecolor=fc
    )
    plt.gca().add_patch(box)
    plt.text(x + w / 2, y + h / 2, text,
             ha="center", va="center", fontsize=11)

# ---------- Canvas ----------
plt.figure(figsize=(10, 7))
plt.axis("off")

# ---------- Colors (pastel) ----------
c_input = "#EEF2F7"
c_t1 = "#E8F5E9"
c_t2 = "#FFF8E1"
c_t3 = "#E3F2FD"
c_output = "#F3E5F5"
c_sub = "#FFFFFF"

# ---------- Layout ----------
x_center = 0.5

# Input
draw_box(0.35, 0.88, 0.3, 0.08,
         "User Query\n+ Candidate KG Paths",
         c_input)

# Tier 1
draw_box(0.25, 0.73, 0.5, 0.1,
         "Tier 1: Symbolic Filter\n"
         "• Type Constraints\n"
         "• Hop Limits\n"
         "• Structural Validity",
         c_t1)

# Tier 2 main
draw_box(0.25, 0.50, 0.5, 0.12,
         "Tier 2: Semantic Scoring\n"
         "Lightweight Similarity Models",
         c_t2)

# Tier 2 sub-boxes
draw_box(0.15, 0.38, 0.2, 0.08,
         "Fast-Pass\n(High Confidence)",
         c_sub)

draw_box(0.40, 0.38, 0.2, 0.08,
         "Ambiguity Zone\n(Selective Escalation)",
         c_sub)

draw_box(0.65, 0.38, 0.2, 0.08,
         "Discard\n(Low Confidence)",
         c_sub)

# Tier 3
draw_box(0.25, 0.22, 0.5, 0.1,
         "Tier 3: LLM Validation\n"
         "Deep Semantic Reasoning\n"
         "(On-demand Only)",
         c_t3)

# Output
draw_box(0.35, 0.08, 0.3, 0.08,
         "Final Answer Path",
         c_output)

# ---------- Arrows ----------
def draw_arrow(x1, y1, x2, y2, rad=0.0):
    style = f"arc3,rad={rad}" if rad else "arc3"
    plt.annotate("",
                 xy=(x2, y2),
                 xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="#444444", connectionstyle=style))

# 1. Input -> Tier 1
# Input bottom: 0.88, Tier 1 top: 0.73 + 0.1 = 0.83
draw_arrow(0.5, 0.88, 0.5, 0.83)

# 2. Tier 1 -> Tier 2 Main
# Tier 1 bottom: 0.73, Tier 2 top: 0.50 + 0.12 = 0.62
draw_arrow(0.5, 0.73, 0.5, 0.62)

# 3. Tier 2 Main -> Leaves (Fast-Pass, Ambiguity, Discard)
# Tier 2 bottom: 0.50
# Leaf tops: 0.38 + 0.08 = 0.46
# Top centers: Fast-Pass(0.25), Ambiguity(0.5), Discard(0.75)

# Center to Fast-Pass (curved left)
draw_arrow(0.5, 0.50, 0.25, 0.46, rad=0.2)
# Center to Ambiguity (straight)
draw_arrow(0.5, 0.50, 0.5, 0.46)
# Center to Discard (curved right)
draw_arrow(0.5, 0.50, 0.75, 0.46, rad=-0.2)

# 4. Ambiguity -> Tier 3
# Ambiguity bottom: 0.38, Tier 3 top: 0.22 + 0.1 = 0.32
draw_arrow(0.5, 0.38, 0.5, 0.32)

# 5. Tier 3 -> Output
# Tier 3 bottom: 0.22, Output top: 0.08 + 0.08 = 0.16
draw_arrow(0.5, 0.22, 0.5, 0.16)

# 6. Fast-Pass -> Output (Direct path)
# Fast-Pass bottom: 0.38. Output top left-ish: let's map to 0.4
# Use a larger curve to bypass Tier 3 visually
draw_arrow(0.25, 0.38, 0.4, 0.16, rad=-0.3)

# ---------- Save ----------
plt.tight_layout()
plt.savefig("tiered_forest_architecture.png", dpi=300, bbox_inches="tight")
plt.show()
