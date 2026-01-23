import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define nodes for each tier
nodes = {
    "Candidate Paths\nfrom KG": (0, 6),
    "Tier 1:\nSymbolic Filter": (0, 4),
    "Pass Tier 1\n→ Tier 2": (-2, 2),
    "Discard Tier 1": (2, 2),
    "Tier 2:\nEmbedding Scoring": (0, 0),
    "Fast-Pass": (-2, -2),
    "Discard Tier 2": (2, -2),
    "Ambiguity Zone\n→ Tier 3": (0, -2),
    "Tier 3:\nLLM Validation": (0, -4),
    "Accepted Path": (-2, -6),
    "Rejected Path": (2, -6),
    "Final Output": (0, -8)
}

# Add nodes
for node, pos in nodes.items():
    G.add_node(node, pos=pos)

# Define edges
edges = [
    ("Candidate Paths\nfrom KG", "Tier 1:\nSymbolic Filter"),
    ("Tier 1:\nSymbolic Filter", "Pass Tier 1\n→ Tier 2"),
    ("Tier 1:\nSymbolic Filter", "Discard Tier 1"),
    ("Pass Tier 1\n→ Tier 2", "Tier 2:\nEmbedding Scoring"),
    ("Tier 2:\nEmbedding Scoring", "Fast-Pass"),
    ("Tier 2:\nEmbedding Scoring", "Discard Tier 2"),
    ("Tier 2:\nEmbedding Scoring", "Ambiguity Zone\n→ Tier 3"),
    ("Ambiguity Zone\n→ Tier 3", "Tier 3:\nLLM Validation"),
    ("Tier 3:\nLLM Validation", "Accepted Path"),
    ("Tier 3:\nLLM Validation", "Rejected Path"),
    ("Fast-Pass", "Final Output"),
    ("Accepted Path", "Final Output")
]

G.add_edges_from(edges)

# Get positions
pos = nx.get_node_attributes(G, 'pos')

# Draw nodes with colors
node_colors = []
for node in G.nodes():
    if "Discard" in node or "Rejected" in node:
        node_colors.append("#f28e8e")  # red for discard
    elif "Fast-Pass" in node or "Accepted" in node or "Final Output" in node:
        node_colors.append("#8fe28e")  # green for accepted
    elif "Ambiguity" in node:
        node_colors.append("#f2d88e")  # orange for escalate
    else:
        node_colors.append("#8ec6f2")  # blue for tiers

# Draw graph
plt.figure(figsize=(8, 10))
nx.draw(G, pos, with_labels=True, node_size=2500, node_color=node_colors, 
        font_size=10, font_weight='bold', arrowsize=20)

plt.title("Tiered-Forest Multi-Level Cascading KG Reasoning", fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig("tiered_forest.png", dpi=300)
plt.show()



