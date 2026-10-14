import json

targets = {'greater', 'equal', 'greater-equal', 'less-equal'}
with open('data/relex/relex_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

extracted = []
for doc in data:
    tokens = doc['tokenized_text']
    rels = doc['relations']
    ner = doc['ner']

    # Filtra frases que contenham as relações numéricas minoritárias
    if any(r[2] in targets for r in rels):
        sentence_str = " ".join(tokens)
        
        entities_list = []
        for s, e, label in ner:
            ent_text = " ".join(tokens[s : e + 1])
            entities_list.append({"text": ent_text, "label": label})

        relations_list = []
        for h_idx, t_idx, r_label in rels:
            h_text = " ".join(tokens[ner[h_idx][0] : ner[h_idx][1] + 1])
            t_text = " ".join(tokens[ner[t_idx][0] : ner[t_idx][1] + 1])
            relations_list.append({"head": h_text, "tail": t_text, "relation": r_label})

        extracted.append({
            "sentence": sentence_str,
            "entities": entities_list,
            "relations": relations_list
        })

        if len(extracted) == 4:
            break

print(json.dumps(extracted, indent=2, ensure_ascii=False))