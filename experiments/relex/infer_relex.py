from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import torch
from gliner import GLiNER

MODEL_PATH = "bornia/gliner-relex-code-accord-augmented"
NER_THRESHOLD = 0.40
REL_THRESHOLD = 0.60
OUTPUT_DIR = Path("outputs/inference_graphs")

# Sentenças para inferência
SENTENCES = [
    "Every fire door must be self-closing and have a clear width of at least 850 mm.",
    "Perimeter insulation should be continuous and have a minimum thickness of 25mm."
]

ENTITIES = ["object", "property", "quality", "value"]
RELATIONS = [
    "equal", "greater", "greater-equal", "less", "less-equal",
    "necessity", "part-of", "selection"
]
COLORS = {
    "object": "#A5D8FF",
    "property": "#FFE066",
    "quality": "#FFC078",
    "value": "#8CE99A"
}


def draw_graph(entities, relations, out_path: Path):
    G = nx.DiGraph()
    for i, e in enumerate(entities):
        G.add_node(i, label=f"{e['text']}\n({e['label']})", color=COLORS.get(e["label"], "#DEE2E6"))

    edge_labels = {}
    for r in relations:
        h = next((i for i, e in enumerate(entities) if e["start"] == r["head"]["start"] and e["end"] == r["head"]["end"]), None)
        t = next((i for i, e in enumerate(entities) if e["start"] == r["tail"]["start"] and e["end"] == r["tail"]["end"]), None)
        if h is not None and t is not None:
            G.add_edge(h, t)
            edge_labels[(h, t)] = r["relation"]

    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42, k=1.4)
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes}

    nx.draw(
        G, pos, ax=ax, labels=node_labels, node_color=node_colors,
        node_size=3400, font_size=8.5, font_weight="bold",
        edge_color="#495057", arrows=True, arrowsize=18
    )
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=8.5, font_color="#212529")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando modelo '{MODEL_PATH}' em {device}...")
    model = GLiNER.from_pretrained(MODEL_PATH).to(device)

    ents_batch, rels_batch = model.inference(
        texts=SENTENCES,
        labels=ENTITIES,
        relations=RELATIONS,
        threshold=NER_THRESHOLD,
        relation_threshold=REL_THRESHOLD,
        return_relations=True,
        flat_ner=False
    )

    for idx, (text, ents, rels) in enumerate(zip(SENTENCES, ents_batch, rels_batch), start=1):
        print(f"\n[{idx}] Sentença: {text}")
        print("Entidades:", [(text[e["start"]:e["end"]], e["label"]) for e in ents])
        print("Relações :", [(text[r["head"]["start"]:r["head"]["end"]], r["relation"], text[r["tail"]["start"]:r["tail"]["end"]]) for r in rels])

        for e in ents:
            e["text"] = text[e["start"]:e["end"]]

        img_file = OUTPUT_DIR / f"sentenca_{idx}.png"
        draw_graph(ents, rels, img_file)
        print(f"Grafo salvo em: {img_file}")


if __name__ == "__main__":
    main()