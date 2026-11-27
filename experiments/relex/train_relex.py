# experiments/relex/train_relex.py
import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from gliner import GLiNER
from gliner.data_processing import (
    RelationExtractionSpanProcessor,
    RelationExtractionSpanDataCollator,
    WordsSplitter
)

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


def evaluate_relex_dataset(
    model,
    json_path: Path,
    batch_size: int = 16,
    ner_threshold: float = 0.3,
    rel_threshold: float = 0.5,
    split_name: str = "TEST"
):
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logger.info(f"Avaliando split {split_name} ({len(dataset)} sentenças)...")

    tp_rel, fp_rel, fn_rel = Counter(), Counter(), Counter()
    tp_ner, fp_ner, fn_ner = Counter(), Counter(), Counter()
    t0 = time.perf_counter()

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Avaliando {split_name}"):
        batch_docs = dataset[i : i + batch_size]
        texts = [" ".join(doc['tokenized_text']) for doc in batch_docs]

        pred_entities_batch, pred_relations_batch = model.inference(
            texts=texts,
            labels=CANONICAL_ENTITIES,
            relations=CANONICAL_RELATIONS,
            threshold=ner_threshold,
            relation_threshold=rel_threshold,
            return_relations=True,
            flat_ner=False
        )

        for doc, pred_ents, pred_rels in zip(batch_docs, pred_entities_batch, pred_relations_batch):
            tokens = doc['tokenized_text']
            char_spans = compute_token_char_spans(tokens)

            # NER Evaluation
            gold_ner_list = []
            for s_tok, e_tok, label in doc['ner']:
                if label in CANONICAL_ENTITIES and e_tok < len(char_spans):
                    gold_ner_list.append((char_spans[s_tok][0], char_spans[e_tok][1], label))

            matched_gold_ents = set()
            for p_ent in pred_ents:
                p_span = (p_ent['start'], p_ent['end'], p_ent['label'])
                match = False
                for g_idx, g_span in enumerate(gold_ner_list):
                    if g_idx in matched_gold_ents:
                        continue
                    if p_span[2] == g_span[2]:
                        inter = max(0, min(p_span[1], g_span[1]) - max(p_span[0], g_span[0]))
                        union = max(p_span[1], g_span[1]) - min(p_span[0], g_span[0])
                        if union > 0 and (inter / union) >= 0.5:
                            tp_ner[p_span[2]] += 1
                            matched_gold_ents.add(g_idx)
                            match = True
                            break
                if not match and p_span[2] in CANONICAL_ENTITIES:
                    fp_ner[p_span[2]] += 1

            for g_idx, g_span in enumerate(gold_ner_list):
                if g_idx not in matched_gold_ents:
                    fn_ner[g_span[2]] += 1

            # Relation Evaluation
            gold_rel_list = []
            for h_idx, t_idx, r_label in doc['relations']:
                if r_label in CANONICAL_RELATIONS and h_idx < len(doc['ner']) and t_idx < len(doc['ner']):
                    h_ent = doc['ner'][h_idx]
                    t_ent = doc['ner'][t_idx]
                    h_char = (char_spans[h_ent[0]][0], char_spans[h_ent[1]][1])
                    t_char = (char_spans[t_ent[0]][0], char_spans[t_ent[1]][1])
                    gold_rel_list.append((h_char, t_char, r_label))

            matched_gold_rels = set()
            for p_rel in pred_rels:
                p_h = (p_rel['head']['start'], p_rel['head']['end'])
                p_t = (p_rel['tail']['start'], p_rel['tail']['end'])
                p_label = p_rel['relation']

                match = False
                for g_idx, (g_h, g_t, g_label) in enumerate(gold_rel_list):
                    if g_idx in matched_gold_rels:
                        continue
                    if p_label == g_label:
                        h_inter = max(0, min(p_h[1], g_h[1]) - max(p_h[0], g_h[0]))
                        t_inter = max(0, min(p_t[1], g_t[1]) - max(p_t[0], g_t[0]))
                        if h_inter > 0 and t_inter > 0:
                            tp_rel[p_label] += 1
                            matched_gold_rels.add(g_idx)
                            match = True
                            break
                if not match and p_label in CANONICAL_RELATIONS:
                    fp_rel[p_label] += 1

            for g_idx, (_, _, g_label) in enumerate(gold_rel_list):
                if g_idx not in matched_gold_rels:
                    fn_rel[g_label] += 1

    elapsed = time.perf_counter() - t0

    def compute_scores(labels, tp, fp, fn):
        per_class = {}
        f1_list = []
        for l in labels:
            t, f_p, f_n = tp[l], fp[l], fn[l]
            p = t / (t + f_p) if (t + f_p) > 0 else 0.0
            r = t / (t + f_n) if (t + f_n) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            support = t + f_n
            per_class[l] = {"precision": p, "recall": r, "f1": f1, "support": support}
            if support > 0:
                f1_list.append(f1)
        macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        tot_tp, tot_fp, tot_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
        micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        return per_class, macro_f1, micro_f1, micro_p, micro_r

    ner_metrics, ner_macro, ner_micro, ner_p, ner_r = compute_scores(CANONICAL_ENTITIES, tp_ner, fp_ner, fn_ner)
    rel_metrics, rel_macro, rel_micro, rel_p, rel_r = compute_scores(CANONICAL_RELATIONS, tp_rel, fp_rel, fn_rel)

    print("\n" + "=" * 75)
    print(f"RELATÓRIO DE AVALIAÇÃO - {split_name} ({len(dataset)} sentenças em {elapsed:.1f}s)")
    print("=" * 75)
    print("\n--- EXTRAÇÃO DE RELAÇÕES (RE) ---")
    print(f"{'Relação':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Suporte':<8}")
    print("-" * 65)
    for r_name, m in rel_metrics.items():
        print(f"{r_name:<16} | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f1']*100:>8.2f}% | {m['support']:>8}")
    print("-" * 65)
    print(f"{'MICRO AVERAGE':<16} | {rel_p*100:>8.2f}% | {rel_r*100:>8.2f}% | {rel_micro*100:>8.2f}% | {sum(tp_rel.values()) + sum(fn_rel.values()):>8}")
    print(f"{'MACRO AVERAGE':<16} | {'-':>9} | {'-':>9} | {rel_macro*100:>8.2f}% |")

    print("\n--- RECONHECIMENTO DE ENTIDADES (NER) ---")
    print(f"{'Entidade':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Suporte':<8}")
    print("-" * 65)
    for e_name, m in ner_metrics.items():
        print(f"{e_name:<16} | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f1']*100:>8.2f}% | {m['support']:>8}")
    print("-" * 65)
    print(f"{'MICRO AVERAGE':<16} | {ner_p*100:>8.2f}% | {ner_r*100:>8.2f}% | {ner_micro*100:>8.2f}% | {sum(tp_ner.values()) + sum(fn_ner.values()):>8}")
    print(f"{'MACRO AVERAGE':<16} | {'-':>9} | {'-':>9} | {ner_macro*100:>8.2f}% |")
    print("=" * 75 + "\n")

    return {
        "relation_macro_f1": rel_macro,
        "relation_micro_f1": rel_micro,
        "ner_macro_f1": ner_macro,
        "ner_micro_f1": ner_micro,
        "relation_details": rel_metrics,
        "ner_details": ner_metrics
    }


def adapt_dataset_for_collator(raw_data, collator):
    """Garante de forma autônoma a compatibilidade de schema com o Collator."""
    candidates = [
        # Formato 1: Tuplas com rótulos canônicos
        lambda d: {
            "tokenized_text": d['tokenized_text'],
            "ner": d['ner'],
            "relations": [r for r in d['relations'] if r[2] in CANONICAL_RELATIONS]
        },
        # Formato 2: Relações com spans absolutos dos tokens
        lambda d: {
            "tokenized_text": d['tokenized_text'],
            "ner": d['ner'],
            "relations": [
                (d['ner'][r[0]][0], d['ner'][r[0]][1], d['ner'][r[1]][0], d['ner'][r[1]][1], r[2])
                for r in d['relations']
                if r[2] in CANONICAL_RELATIONS and max(r[0], r[1]) < len(d['ner'])
            ]
        },
        # Formato 3: Dicionário semântico
        lambda d: {
            "tokenized_text": d['tokenized_text'],
            "ner": d['ner'],
            "relations": [
                {"head": d['ner'][r[0]][:2], "tail": d['ner'][r[1]][:2], "relation_text": r[2]}
                for r in d['relations']
                if r[2] in CANONICAL_RELATIONS and max(r[0], r[1]) < len(d['ner'])
            ]
        }
    ]

    for candidate_fn in candidates:
        try:
            test_batch = [candidate_fn(doc) for doc in raw_data[:2]]
            _ = collator(test_batch)
            logger.info("Formato de batch validado com sucesso pelo Collator!")
            return [candidate_fn(doc) for doc in raw_data]
        except Exception:
            continue

    return raw_data


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning conjunto GLiNER-ReLEx.")
    parser.add_argument("--model_path", type=str, default="models/gliner_relex_large")
    parser.add_argument("--train_path", type=str, default="data/relex/relex_train.json")
    parser.add_argument("--val_path", type=str, default="data/relex/relex_val.json")
    parser.add_argument("--test_path", type=str, default="data/relex/relex_test.json")
    parser.add_argument("--output_dir", type=str, default="outputs/relex_model_best")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo de execução: {device.upper()}")

    # 1. Carrega modelo base
    logger.info(f"Carregando GLiNER-ReLEx: {args.model_path}...")
    model = GLiNER.from_pretrained(args.model_path).to(device)

    if args.eval_only:
        _ = evaluate_relex_dataset(model, Path(args.test_path), split_name="TEST (EVAL ONLY)")
        return

    # 2. Configura Processor e Collator nativos de Relações
    logger.info("Configurando RelationExtraction DataCollator nativo...")
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None and hasattr(model, 'data_processor') and hasattr(model.data_processor, 'transformer_tokenizer'):
        tokenizer = model.data_processor.transformer_tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    splitter_type = getattr(model.config, 'words_splitter_type', 'whitespace')
    words_splitter = WordsSplitter(splitter_type)

    data_processor = RelationExtractionSpanProcessor(
        config=model.config,
        tokenizer=tokenizer,
        words_splitter=words_splitter
    )

    data_collator = RelationExtractionSpanDataCollator(
        config=model.config,
        data_processor=data_processor,
        prepare_labels=True
    )

    # 3. Carrega e formata datasets
    with open(args.train_path, 'r', encoding='utf-8') as f:
        train_raw = json.load(f)

    train_data = adapt_dataset_for_collator(train_raw, data_collator)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    # 4. Otimizador e Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    best_val_f1 = -1.0
    best_ckpt_dir = output_dir / "best_checkpoint"

    logger.info(f"Iniciando treino por {args.epochs} épocas ({len(train_loader)} batches/época)...")

    # 5. Loop de Treino com cálculo real de tensores
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Época {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)

            loss = outputs.loss if hasattr(outputs, "loss") else (outputs[0] if isinstance(outputs, (tuple, list)) else outputs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Época {epoch} concluída. Loss média: {avg_loss:.4f}")

        # Avaliação de validação ao final de cada época
        val_res = evaluate_relex_dataset(
            model=model,
            json_path=Path(args.val_path),
            split_name=f"VAL (ÉPOCA {epoch})"
        )

        val_f1 = val_res['relation_macro_f1']
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            logger.info(f"Recorde em Val Macro F1: {val_f1*100:.2f}%. Salvando melhor checkpoint...")
            model.save_pretrained(str(best_ckpt_dir))

    # 6. Avaliação Final do Test Set Completo com o Melhor Checkpoint
    logger.info("Carregando o melhor checkpoint para a avaliação final do Test Set...")
    best_model = GLiNER.from_pretrained(str(best_ckpt_dir)).to(device)

    final_test_results = evaluate_relex_dataset(
        model=best_model,
        json_path=Path(args.test_path),
        split_name="TEST FINAL (FINE-TUNED)"
    )

    out_file = output_dir / "final_fine_tuned_test_results.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_test_results, f, indent=2)

    logger.info(f"Resultados consolidados exportados em: {out_file.resolve()}")


if __name__ == '__main__':
    main()