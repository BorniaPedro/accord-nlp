# experiments/relex/debug_inspect.py
import json
from gliner import GLiNER

CANONICAL_ENTITIES = ['object', 'property', 'quality', 'value']
CANONICAL_RELATIONS = [
    'equal', 'greater', 'greater-equal', 'less', 'less-equal',
    'necessity', 'part-of', 'selection'
]

model = GLiNER.from_pretrained("outputs/relex_model_best/best_checkpoint")
dataset = json.load(open("data/relex/relex_test.json", encoding="utf-8"))

print("=" * 80)
print("AUDITORIA QUALITATIVA: GABARITO (GOLD) VS PREDIÇÃO REAL")
print("=" * 80)

for idx in range(3):
    doc = dataset[idx]
    tokens = doc['tokenized_text']
    text = " ".join(tokens)

    # Predição direta
    ents, rels = model.inference(
        texts=[text],
        labels=CANONICAL_ENTITIES,
        relations=CANONICAL_RELATIONS,
        threshold=0.3,
        relation_threshold=0.5,
        return_relations=True
    )

    print(f"\n[SENTENÇA {idx + 1}]:\n\"{text[:100]}...\"\n")

    # Mapeamento do Gabarito
    print("  --- GABARITO (GOLD) ---")
    gold_ents = [(tokens[s:e+1], label) for s, e, label in doc['ner']]
    for h_idx, t_idx, r_label in doc['relations']:
        h = " ".join(tokens[doc['ner'][h_idx][0] : doc['ner'][h_idx][1] + 1])
        t = " ".join(tokens[doc['ner'][t_idx][0] : doc['ner'][t_idx][1] + 1])
        print(f"  [GOLD]  ('{h}') --[{r_label}]--> ('{t}')")

    print("\n  --- PREDIÇÃO DO MODELO ---")
    if not rels[0]:
        print("  [PRED]  Nenhuma relação detectada.")
    for r in rels[0]:
        print(f"  [PRED]  ('{r['head']['text']}') --[{r['relation']}]--> ('{r['tail']['text']}') | score: {r['score']:.2f}")

    print("-" * 80)