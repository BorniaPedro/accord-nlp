# experiments/relex/eval_strict.py
import json
import logging
import time
from collections import Counter
from pathlib import Path
import torch
from tqdm import tqdm
from gliner import GLiNER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_ENTITIES = ['object', 'property', 'quality', 'value']
CANONICAL_RELATIONS = [
    'equal', 'greater', 'greater-equal', 'less', 'less-equal',
    'necessity', 'part-of', 'selection'
]


def compute_token_char_spans(tokens):
    spans = []
    pos = 0
    for t in tokens:
        spans.append((pos, pos + len(t)))
        pos += len(t) + 1
    return spans


def run_strict_evaluation(
    model_path="outputs/relex_model_best/best_checkpoint",
    test_path="data/relex/relex_test.json",
    batch_size=16
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Carregando checkpoint de: {model_path}...")
    model = GLiNER.from_pretrained(model_path).to(device)

    with open(test_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logger.info(f"Avaliando Test Set ({len(dataset)} sentenças) sob Critério Estrito...")

    # Contadores Estritos (Strict Match: limites exatos + labels corretos)
    tp_strict_rel = Counter()
    fp_strict_rel = Counter()
    fn_strict_rel = Counter()

    # Contadores Estritos de NER
    tp_strict_ner = Counter()
    fp_strict_ner = Counter()
    fn_strict_ner = Counter()

    t0 = time.perf_counter()

    for i in tqdm(range(0, len(dataset), batch_size), desc="Avaliando"):
        batch_docs = dataset[i : i + batch_size]
        texts = [" ".join(doc['tokenized_text']) for doc in batch_docs]

        pred_entities_batch, pred_relations_batch = model.inference(
            texts=texts,
            labels=CANONICAL_ENTITIES,
            relations=CANONICAL_RELATIONS,
            threshold=0.3,
            relation_threshold=0.5,
            return_relations=True,
            flat_ner=False
        )

        for doc, pred_ents, pred_rels in zip(batch_docs, pred_entities_batch, pred_relations_batch):
            tokens = doc['tokenized_text']
            char_spans = compute_token_char_spans(tokens)

            # -------------------------------------------------------------
            # 1. GABARITO (GOLD)
            # -------------------------------------------------------------
            gold_ner_map = {}  # idx -> (start, end, label)
            for idx, (s_tok, e_tok, label) in enumerate(doc['ner']):
                if label in CANONICAL_ENTITIES and e_tok < len(char_spans):
                    gold_ner_map[idx] = (char_spans[s_tok][0], char_spans[e_tok][1], label)

            gold_rel_list = []
            for h_idx, t_idx, r_label in doc['relations']:
                if r_label in CANONICAL_RELATIONS and h_idx in gold_ner_map and t_idx in gold_ner_map:
                    h_span = gold_ner_map[h_idx]
                    t_span = gold_ner_map[t_idx]
                    # Tripla gold completa: ((h_start, h_end, h_label), (t_start, t_end, t_label), rel_label)
                    gold_rel_list.append((h_span, t_span, r_label))

            # -------------------------------------------------------------
            # 2. NER STRICT EVALUATION
            # -------------------------------------------------------------
            gold_ents_strict = list(gold_ner_map.values())
            matched_gold_ents = set()

            for p_ent in pred_ents:
                p_span = (p_ent['start'], p_ent['end'], p_ent['label'])
                match = False
                for g_idx, g_span in enumerate(gold_ents_strict):
                    if g_idx in matched_gold_ents:
                        continue
                    # STRICT MATCH: Exatamente mesmo início, mesmo fim e mesma classe
                    if p_span == g_span:
                        tp_strict_ner[p_span[2]] += 1
                        matched_gold_ents.add(g_idx)
                        match = True
                        break
                if not match and p_span[2] in CANONICAL_ENTITIES:
                    fp_strict_ner[p_span[2]] += 1

            for g_idx, g_span in enumerate(gold_ents_strict):
                if g_idx not in matched_gold_ents:
                    fn_strict_ner[g_span[2]] += 1

            # -------------------------------------------------------------
            # 3. RE STRICT EVALUATION (E2E PIPELINE)
            # -------------------------------------------------------------
            # -------------------------------------------------------------
            # RE STRICT TOTAL (Span Exato + Entidade Exata + Relação Exata)
            # -------------------------------------------------------------
            matched_gold_rels = set()

            for p_rel in pred_rels:
                p_h_start, p_h_end = p_rel['head']['start'], p_rel['head']['end']
                p_t_start, p_t_end = p_rel['tail']['start'], p_rel['tail']['end']
                p_label = p_rel['relation']

                # 1. Recupera o tipo de entidade que o próprio modelo previu no NER
                pred_h_type = None
                pred_t_type = None
                for pe in pred_ents:
                    if pe['start'] == p_h_start and pe['end'] == p_h_end:
                        pred_h_type = pe['label']
                    if pe['start'] == p_t_start and pe['end'] == p_t_end:
                        pred_t_type = pe['label']

                match = False
                for g_idx, (g_h, g_t, g_rel) in enumerate(gold_rel_list):
                    if g_idx in matched_gold_rels:
                        continue

                    # Tolerância ZERO de caracteres e igualdade estrita de classes
                    h_exact = (p_h_start == g_h[0] and p_h_end == g_h[1])
                    t_exact = (p_t_start == g_t[0] and p_t_end == g_t[1])
                    rel_exact = (p_label == g_rel)
                    types_exact = (pred_h_type == g_h[2] and pred_t_type == g_t[2])

                    if h_exact and t_exact and rel_exact and types_exact:
                        tp_strict_rel[p_label] += 1
                        matched_gold_rels.add(g_idx)
                        match = True
                        break

                if not match and p_label in CANONICAL_RELATIONS:
                    fp_strict_rel[p_label] += 1

            for g_idx, (_, _, g_rel) in enumerate(gold_rel_list):
                if g_idx not in matched_gold_rels:
                    fn_strict_rel[g_rel] += 1

    elapsed = time.perf_counter() - t0

    # Cálculo métricas
    def calc_metrics(labels, tp, fp, fn):
        per_class = {}
        f1_list = []
        for l in labels:
            t = tp[l]
            f_p = fp[l]
            f_n = fn[l]
            p = t / (t + f_p) if (t + f_p) > 0 else 0.0
            r = t / (t + f_n) if (t + f_n) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            per_class[l] = (p, r, f1, t + f_n)
            if (t + f_n) > 0:
                f1_list.append(f1)
        macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        tot_tp, tot_fp, tot_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
        micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        return per_class, macro_f1, micro_f1, micro_p, micro_r

    r_cls, r_macro, r_micro, r_p, r_r = calc_metrics(CANONICAL_RELATIONS, tp_strict_rel, fp_strict_rel, fn_strict_rel)
    n_cls, n_macro, n_micro, n_p, n_r = calc_metrics(CANONICAL_ENTITIES, tp_strict_ner, fp_strict_ner, fn_strict_ner)

    print("\n" + "=" * 75)
    print(f"AVALIAÇÃO ESTRITA OFICIAL (STRICT MATCH) - TEST SET ({elapsed:.1f}s)")
    print("=" * 75)
    print("\n--- EXTRAÇÃO DE RELAÇÕES (RE STRICT) ---")
    print(f"{'Relação':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Suporte':<8}")
    print("-" * 65)
    for r_name, (p, r, f1, sup) in r_cls.items():
        print(f"{r_name:<16} | {p*100:>8.2f}% | {r*100:>8.2f}% | {f1*100:>8.2f}% | {sup:>8}")
    print("-" * 65)
    print(f"{'MACRO AVERAGE':<16} | {'-':>9} | {'-':>9} | {r_macro*100:>8.2f}% |")
    print(f"{'MICRO AVERAGE':<16} | {r_p*100:>8.2f}% | {r_r*100:>8.2f}% | {r_micro*100:>8.2f}% |")

    print("\n--- RECONHECIMENTO DE ENTIDADES (NER STRICT) ---")
    print(f"{'Entidade':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Suporte':<8}")
    print("-" * 65)
    for e_name, (p, r, f1, sup) in n_cls.items():
        print(f"{e_name:<16} | {p*100:>8.2f}% | {r*100:>8.2f}% | {f1*100:>8.2f}% | {sup:>8}")
    print("-" * 65)
    print(f"{'MACRO AVERAGE':<16} | {'-':>9} | {'-':>9} | {n_macro*100:>8.2f}% |")
    print(f"{'MICRO AVERAGE':<16} | {n_p*100:>8.2f}% | {n_r*100:>8.2f}% | {n_micro*100:>8.2f}% |")
    print("=" * 75)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Avaliação Estrita GLiNER-ReLEx")
    parser.add_argument("--model_path", type=str, default="outputs/relex_model_best/best_checkpoint")
    parser.add_argument("--test_path", type=str, default="data/relex/relex_test.json")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run_strict_evaluation(
        model_path=args.model_path,
        test_path=args.test_path,
        batch_size=args.batch_size
    )