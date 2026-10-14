# data_processing/build_relex_dataset.py
import argparse
import json
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def extract_entities_from_bio(tokens, bio_tags):
    entities = []
    i = 0
    n = len(tokens)
    while i < n:
        tag = bio_tags[i]
        if tag.startswith('B-'):
            ent_label = tag[2:]
            start_idx = i
            end_idx = i
            i += 1
            while i < n and bio_tags[i] == f'I-{ent_label}':
                end_idx = i
                i += 1
            entities.append([start_idx, end_idx, ent_label])
        else:
            i += 1
    return entities


def locate_spans_in_tokens(clean_tokens, tagged_sentence):
    m1 = re.search(r'<e1>(.*?)</e1>', tagged_sentence)
    m2 = re.search(r'<e2>(.*?)</e2>', tagged_sentence)
    if not m1 or not m2:
        return None, None

    clean_text = re.sub(r'</?e[12]>', '', tagged_sentence).strip()
    e1_start = len(re.sub(r'</?e[12]>', '', tagged_sentence[:m1.start()]))
    e1_end = e1_start + len(m1.group(1).strip())
    e2_start = len(re.sub(r'</?e[12]>', '', tagged_sentence[:m2.start()]))
    e2_end = e2_start + len(m2.group(1).strip())

    tok_spans = []
    pos = 0
    for t in clean_tokens:
        p = clean_text.find(t, pos)
        if p == -1:
            p = pos
        tok_spans.append((p, p + len(t)))
        pos = p + len(t)

    def to_token_range(s, e):
        idx = [i for i, (ts, te) in enumerate(tok_spans) if not (te <= s or ts >= e)]
        return (idx[0], idx[-1]) if idx else None

    return to_token_range(e1_start, e1_end), to_token_range(e2_start, e2_end)


def convert_df_to_relex(ner_df, all_re_df):
    rel_map = {}
    for _, r in all_re_df.iterrows():
        eid = r['example_id']
        if r['relation_type'] != 'none' and pd.notna(r['relation_type']):
            rel_map.setdefault(eid, []).append((r['tagged_sentence'], r['relation_type']))

    records = []
    total_rel = 0
    for _, row in ner_df.iterrows():
        eid = row['example_id']
        tokens = row['processed_content'].strip().split()
        bio_tags = row['label'].strip().split()
        min_len = min(len(tokens), len(bio_tags))
        tokens = tokens[:min_len]
        bio_tags = bio_tags[:min_len]

        ner_spans = extract_entities_from_bio(tokens, bio_tags)
        relations = []
        seen_triplets = set()  # Garante unicidade das arestas no grafo

        if eid in rel_map:
            for tagged_s, r_type in rel_map[eid]:
                e1_span, e2_span = locate_spans_in_tokens(tokens, tagged_s)
                if not e1_span or not e2_span:
                    continue

                def match_ner(span):
                    # 1. Prioriza casamento exato de limites
                    for idx, (s, e, _) in enumerate(ner_spans):
                        if s == span[0] and e == span[1]:
                            return idx
                    # 2. Fallback por sobreposição
                    for idx, (s, e, _) in enumerate(ner_spans):
                        if not (e < span[0] or s > span[1]):
                            return idx
                    return None

                h = match_ner(e1_span)
                t = match_ner(e2_span)
                if h is not None and t is not None and h != t:
                    triplet_key = (h, t, r_type)
                    if triplet_key not in seen_triplets:
                        seen_triplets.add(triplet_key)
                        relations.append([h, t, r_type])
                        total_rel += 1

        records.append({
            "tokenized_text": tokens,
            "ner": ner_spans,
            "relations": relations
        })
    return records, total_rel


def main():
    os.makedirs("data/relex", exist_ok=True)
    
    # 1. Carrega os arquivos CSV locais
    data_dir = "data/all_csv"
    train_ner = pd.read_csv(f"{data_dir}/train_ner.csv")
    test_ner = pd.read_csv(f"{data_dir}/test_ner.csv")
    train_re = pd.read_csv(f"{data_dir}/train_re.csv")
    test_re = pd.read_csv(f"{data_dir}/test_re.csv")
    all_re = pd.concat([train_re, test_re], ignore_index=True)

    # 2. Divide o treino (689 sentenças) em treino (620) e validação (69) com semente 42
    train_df, val_df = train_test_split(train_ner, test_size=0.10, random_state=42)

    # 3. Converte os splits preservando a integridade das sentenças
    train_relex, tr_r = convert_df_to_relex(train_df, all_re)
    val_relex, val_r = convert_df_to_relex(val_df, all_re)
    test_relex, te_r = convert_df_to_relex(test_ner, all_re)

    print(f"Treino : {len(train_relex)} sentenças | {tr_r} relações")
    print(f"Val    : {len(val_relex)} sentenças | {val_r} relações")
    print(f"Teste  : {len(test_relex)} sentenças | {te_r} relações")

    with open("data/relex/relex_train.json", "w", encoding="utf-8") as f:
        json.dump(train_relex, f, indent=2, ensure_ascii=False)
    with open("data/relex/relex_val.json", "w", encoding="utf-8") as f:
        json.dump(val_relex, f, indent=2, ensure_ascii=False)
    with open("data/relex/relex_test.json", "w", encoding="utf-8") as f:
        json.dump(test_relex, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()