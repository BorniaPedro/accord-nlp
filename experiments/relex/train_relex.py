import argparse
import csv
import itertools
import json
import logging
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from gliner import GLiNER
from gliner.data_processing import (
    RelationExtractionSpanDataCollator,
    RelationExtractionSpanProcessor,
    WordsSplitter,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_ENTITIES = ["object", "property", "quality", "value"]
CANONICAL_RELATIONS = [
    "equal", "greater", "greater-equal", "less", "less-equal",
    "necessity", "part-of", "selection"
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_token_char_spans(tokens: list[str]) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    for t in tokens:
        spans.append((pos, pos + len(t)))
        pos += len(t) + 1
    return spans


def check_overlap(span1: tuple[int, int], span2: tuple[int, int]) -> bool:
    return max(span1[0], span2[0]) < min(span1[1], span2[1])


def evaluate_relex_dataset(
    model,
    json_path: Path,
    batch_size: int = 16,
    ner_threshold: float = 0.40,
    rel_threshold: float = 0.50,
    split_name: str = "TEST",
    silent: bool = False,
    collect_predictions: bool = False
):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tp_ner_strict, fp_ner_strict, fn_ner_strict = Counter(), Counter(), Counter()
    tp_rel_strict, fp_rel_strict, fn_rel_strict = Counter(), Counter(), Counter()
    tp_ner_relax, fp_ner_relax, fn_ner_relax = Counter(), Counter(), Counter()
    tp_rel_relax, fp_rel_relax, fn_rel_relax = Counter(), Counter(), Counter()
    predictions_log = []

    t0 = time.perf_counter()
    iterator = range(0, len(dataset), batch_size)
    if not silent:
        iterator = tqdm(iterator, desc=f"Avaliando {split_name}")

    doc_offset = 0
    for i in iterator:
        batch_docs = dataset[i : i + batch_size]
        texts = [" ".join(doc["tokenized_text"]) for doc in batch_docs]

        pred_entities_batch, pred_relations_batch = model.inference(
            texts=texts,
            labels=CANONICAL_ENTITIES,
            relations=CANONICAL_RELATIONS,
            threshold=ner_threshold,
            relation_threshold=rel_threshold,
            return_relations=True,
            flat_ner=False
        )

        for local_idx, (doc, text, pred_ents, pred_rels) in enumerate(
            zip(batch_docs, texts, pred_entities_batch, pred_relations_batch)
        ):
            tokens = doc["tokenized_text"]
            char_spans = compute_token_char_spans(tokens)

            gold_ner_list = []
            for s_tok, e_tok, label in doc["ner"]:
                if label in CANONICAL_ENTITIES and e_tok < len(char_spans):
                    gold_ner_list.append((char_spans[s_tok][0], char_spans[e_tok][1], label))

            gold_rel_list = []
            for h_idx, t_idx, r_label in doc["relations"]:
                if r_label in CANONICAL_RELATIONS and h_idx < len(doc["ner"]) and t_idx < len(doc["ner"]):
                    h_ent = doc["ner"][h_idx]
                    t_ent = doc["ner"][t_idx]
                    h_char = (char_spans[h_ent[0]][0], char_spans[h_ent[1]][1])
                    t_char = (char_spans[t_ent[0]][0], char_spans[t_ent[1]][1])
                    gold_rel_list.append((h_char, t_char, r_label))

            if collect_predictions:
                predictions_log.append({
                    "doc_id": doc_offset + local_idx,
                    "sentence": text,
                    "gold_entities": [
                        {"text": text[s:e], "label": lbl, "char_span": [s, e]}
                        for s, e, lbl in gold_ner_list
                    ],
                    "pred_entities": [
                        {"text": text[p["start"]:p["end"]], "label": p["label"], "char_span": [p["start"], p["end"]]}
                        for p in pred_ents
                    ],
                    "gold_relations": [
                        {"head": text[gh[0]:gh[1]], "relation": gl, "tail": text[gt[0]:gt[1]]}
                        for gh, gt, gl in gold_rel_list
                    ],
                    "pred_relations": [
                        {
                            "head": text[p["head"]["start"]:p["head"]["end"]],
                            "relation": p["relation"],
                            "tail": text[p["tail"]["start"]:p["tail"]["end"]]
                        }
                        for p in pred_rels
                    ]
                })

            # 1. NER (Estrito e Relaxado)
            matched_gold_ner_strict = set()
            matched_gold_ner_relax = set()

            for p_ent in pred_ents:
                p_span = (p_ent["start"], p_ent["end"], p_ent["label"])

                for g_idx, g_span in enumerate(gold_ner_list):
                    if g_idx not in matched_gold_ner_strict and p_span == g_span:
                        tp_ner_strict[p_span[2]] += 1
                        matched_gold_ner_strict.add(g_idx)
                        break
                else:
                    if p_span[2] in CANONICAL_ENTITIES:
                        fp_ner_strict[p_span[2]] += 1

                for g_idx, g_span in enumerate(gold_ner_list):
                    if (
                        g_idx not in matched_gold_ner_relax
                        and p_span[2] == g_span[2]
                        and check_overlap(p_span[:2], g_span[:2])
                    ):
                        tp_ner_relax[p_span[2]] += 1
                        matched_gold_ner_relax.add(g_idx)
                        break
                else:
                    if p_span[2] in CANONICAL_ENTITIES:
                        fp_ner_relax[p_span[2]] += 1

            for g_idx, g_span in enumerate(gold_ner_list):
                if g_idx not in matched_gold_ner_strict:
                    fn_ner_strict[g_span[2]] += 1
                if g_idx not in matched_gold_ner_relax:
                    fn_ner_relax[g_span[2]] += 1

            # 2. RE (Estrito e Relaxado)
            matched_gold_rel_strict = set()
            matched_gold_rel_relax = set()

            for p_rel in pred_rels:
                p_h = (p_rel["head"]["start"], p_rel["head"]["end"])
                p_t = (p_rel["tail"]["start"], p_rel["tail"]["end"])
                p_label = p_rel["relation"]

                for g_idx, (g_h, g_t, g_label) in enumerate(gold_rel_list):
                    if g_idx not in matched_gold_rel_strict and p_label == g_label and p_h == g_h and p_t == g_t:
                        tp_rel_strict[p_label] += 1
                        matched_gold_rel_strict.add(g_idx)
                        break
                else:
                    if p_label in CANONICAL_RELATIONS:
                        fp_rel_strict[p_label] += 1

                for g_idx, (g_h, g_t, g_label) in enumerate(gold_rel_list):
                    if (
                        g_idx not in matched_gold_rel_relax
                        and p_label == g_label
                        and check_overlap(p_h, g_h)
                        and check_overlap(p_t, g_t)
                    ):
                        tp_rel_relax[p_label] += 1
                        matched_gold_rel_relax.add(g_idx)
                        break
                else:
                    if p_label in CANONICAL_RELATIONS:
                        fp_rel_relax[p_label] += 1

            for g_idx, (_, _, g_label) in enumerate(gold_rel_list):
                if g_idx not in matched_gold_rel_strict:
                    fn_rel_strict[g_label] += 1
                if g_idx not in matched_gold_rel_relax:
                    fn_rel_relax[g_label] += 1

        doc_offset += len(batch_docs)

    elapsed = time.perf_counter() - t0

    def compute_scores(labels, tp, fp, fn):
        f1_list = []
        for l in labels:
            t, f_p, f_n = tp[l], fp[l], fn[l]
            p = t / (t + f_p) if (t + f_p) > 0 else 0.0
            r = t / (t + f_n) if (t + f_n) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            if (t + f_n) > 0:
                f1_list.append(f1)
        macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        tot_tp, tot_fp, tot_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
        micro_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        micro_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        return macro_f1, micro_f1

    def per_class_breakdown(labels, tp_s, fp_s, fn_s, tp_r, fp_r, fn_r):
        rows = {}
        for l in labels:
            ts, fps, fns = tp_s[l], fp_s[l], fn_s[l]
            ps = ts / (ts + fps) if (ts + fps) > 0 else 0.0
            rs = ts / (ts + fns) if (ts + fns) > 0 else 0.0
            f1s = (2 * ps * rs) / (ps + rs) if (ps + rs) > 0 else 0.0

            tr, fpr, fnr = tp_r[l], fp_r[l], fn_r[l]
            pr = tr / (tr + fpr) if (tr + fpr) > 0 else 0.0
            rr = tr / (tr + fnr) if (tr + fnr) > 0 else 0.0
            f1r = (2 * pr * rr) / (pr + rr) if (pr + rr) > 0 else 0.0

            rows[l] = {
                "support": ts + fns,
                "P_strict": round(ps * 100, 2),
                "R_strict": round(rs * 100, 2),
                "F1_strict": round(f1s * 100, 2),
                "P_relax": round(pr * 100, 2),
                "R_relax": round(rr * 100, 2),
                "F1_relax": round(f1r * 100, 2),
            }
        return rows

    ner_macro_strict, ner_micro_strict = compute_scores(CANONICAL_ENTITIES, tp_ner_strict, fp_ner_strict, fn_ner_strict)
    rel_macro_strict, rel_micro_strict = compute_scores(CANONICAL_RELATIONS, tp_rel_strict, fp_rel_strict, fn_rel_strict)
    ner_macro_relax, ner_micro_relax = compute_scores(CANONICAL_ENTITIES, tp_ner_relax, fp_ner_relax, fn_ner_relax)
    rel_macro_relax, rel_micro_relax = compute_scores(CANONICAL_RELATIONS, tp_rel_relax, fp_rel_relax, fn_rel_relax)

    if not silent:
        print(
            f"\n[{split_name} | {elapsed:.1f}s] NER Strict Micro: {ner_micro_strict*100:.2f}% | "
            f"NER Relax Micro: {ner_micro_relax*100:.2f}% | RE Strict Macro: {rel_macro_strict*100:.2f}% | "
            f"RE Relax Macro: {rel_macro_relax*100:.2f}%"
        )

    result = {
        "rel_macro_strict": rel_macro_strict,
        "rel_micro_strict": rel_micro_strict,
        "ner_macro_strict": ner_macro_strict,
        "ner_micro_strict": ner_micro_strict,
        "rel_macro_relax": rel_macro_relax,
        "rel_micro_relax": rel_micro_relax,
        "ner_macro_relax": ner_macro_relax,
        "ner_micro_relax": ner_micro_relax,
        "ner_per_class": per_class_breakdown(
            CANONICAL_ENTITIES, tp_ner_strict, fp_ner_strict, fn_ner_strict, tp_ner_relax, fp_ner_relax, fn_ner_relax
        ),
        "rel_per_class": per_class_breakdown(
            CANONICAL_RELATIONS, tp_rel_strict, fp_rel_strict, fn_rel_strict, tp_rel_relax, fp_rel_relax, fn_rel_relax
        ),
    }
    if collect_predictions:
        result["predictions"] = predictions_log
    return result


def adapt_dataset_for_collator(raw_data, collator):
    candidates = [
        lambda d: {
            "tokenized_text": d["tokenized_text"],
            "ner": d["ner"],
            "relations": [r for r in d["relations"] if r[2] in CANONICAL_RELATIONS]
        },
        lambda d: {
            "tokenized_text": d["tokenized_text"],
            "ner": d["ner"],
            "relations": [
                (d["ner"][r[0]][0], d["ner"][r[0]][1], d["ner"][r[1]][0], d["ner"][r[1]][1], r[2])
                for r in d["relations"]
                if r[2] in CANONICAL_RELATIONS and max(r[0], r[1]) < len(d["ner"])
            ]
        }
    ]
    for candidate_fn in candidates:
        try:
            _ = collator([candidate_fn(doc) for doc in raw_data[:2]])
            return [candidate_fn(doc) for doc in raw_data]
        except Exception:
            continue
    return raw_data


def build_differential_optimizer(model, lr_encoder, lr_others, weight_decay):
    encoder_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ["token_rep_layer", "bert", "deberta", "encoder"]):
            encoder_params.append(param)
        else:
            other_params.append(param)
    return AdamW([
        {"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay},
        {"params": other_params, "lr": lr_others, "weight_decay": weight_decay}
    ])


def append_to_csv(csv_path, row_dict):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def get_compute_device():
    if torch.cuda.is_available():
        logger.info(f"Aceleração GPU detectada (CUDA/ROCm): {torch.cuda.get_device_name(0)}")
        return torch.device("cuda"), "cuda"
    logger.warning("Nenhuma GPU detectada. Executando em CPU.")
    return torch.device("cpu"), "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning e avaliação GLiNER-ReLEx no CODE-ACCORD.")
    parser.add_argument("--model_path", type=str, default="knowledgator/gliner-relex-large-v0.5")
    parser.add_argument("--train_path", type=str, default="data/relex/relex_train.json")
    parser.add_argument("--val_path", type=str, default="data/relex/relex_val.json")
    parser.add_argument("--test_path", type=str, default="data/relex/relex_test.json")
    parser.add_argument("--output_dir", type=str, default="outputs/relex_model_best")
    parser.add_argument("--csv_out", type=str, default="outputs/benchmark_results_full_grid.csv")

    # Hiperparâmetros
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--others_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_width", type=int, default=12)
    parser.add_argument("--fp16", action="store_true")

    # Limiares de avaliação
    parser.add_argument("--ner_thresholds", nargs="+", type=float, default=[0.30, 0.35, 0.40])
    parser.add_argument("--rel_thresholds", nargs="+", type=float, default=[0.40, 0.50, 0.60])

    # Controle de saída
    parser.add_argument("--no_save_ckpt", action="store_true")
    parser.add_argument("--save_predictions", action="store_true", help="Salva relatório por classe e predições por frase")
    parser.add_argument("--run_name", type=str, default="GLiNER_ReLEx_Run")
    parser.add_argument("--augmented_data", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, backend = get_compute_device()
    model = GLiNER.from_pretrained(args.model_path).to(device)
    model.config.max_width = args.max_width

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None and hasattr(model, "data_processor"):
        tokenizer = model.data_processor.transformer_tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    words_splitter = WordsSplitter(getattr(model.config, "words_splitter_type", "whitespace"))
    data_processor = RelationExtractionSpanProcessor(config=model.config, tokenizer=tokenizer, words_splitter=words_splitter)
    data_collator = RelationExtractionSpanDataCollator(config=model.config, data_processor=data_processor, prepare_labels=True)

    with open(args.train_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    train_data = adapt_dataset_for_collator(train_raw, data_collator)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)

    optimizer = build_differential_optimizer(model, args.lr, args.others_lr, args.weight_decay)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )

    use_amp = args.fp16 and backend == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_f1 = -1.0
    best_epoch = 0
    best_ckpt_dir = output_dir / "best_checkpoint"

    mid_ner_th = args.ner_thresholds[len(args.ner_thresholds) // 2]
    mid_rel_th = args.rel_thresholds[len(args.rel_thresholds) // 2]

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Época {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else (outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
                loss = loss / args.grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}"})

        val_res = evaluate_relex_dataset(
            model, Path(args.val_path),
            batch_size=args.batch_size,
            ner_threshold=mid_ner_th,
            rel_threshold=mid_rel_th,
            split_name=f"VAL EP {epoch}",
            silent=True
        )
        if val_res["rel_macro_strict"] > best_val_f1:
            best_val_f1 = val_res["rel_macro_strict"]
            best_epoch = epoch
            model.to("cpu").save_pretrained(str(best_ckpt_dir))
            model.to(device)

    best_model = GLiNER.from_pretrained(str(best_ckpt_dir)).to(device)
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Avaliando Best Checkpoint (Época {best_epoch})...")
    for ner_th, rel_th in itertools.product(args.ner_thresholds, args.rel_thresholds):
        val_th_res = evaluate_relex_dataset(
            best_model, Path(args.val_path),
            batch_size=args.batch_size,
            ner_threshold=ner_th,
            rel_threshold=rel_th,
            split_name="VAL_TH",
            silent=True
        )
        test_th_res = evaluate_relex_dataset(
            best_model, Path(args.test_path),
            batch_size=args.batch_size,
            ner_threshold=ner_th,
            rel_threshold=rel_th,
            split_name=f"{args.run_name}_ner{ner_th}_rel{rel_th}",
            silent=not args.save_predictions,
            collect_predictions=args.save_predictions
        )

        row = {
            "Run_Name": f"{args.run_name}_ner{ner_th}_rel{rel_th}",
            "Augmented_Data": args.augmented_data,
            "Epochs": args.epochs,
            "Best_Epoch": best_epoch,
            "Batch_Size": args.batch_size,
            "LR_Encoder": args.lr,
            "LR_Others": args.others_lr,
            "NER_Threshold": ner_th,
            "REL_Threshold": rel_th,
            "Val_RE_Strict_Macro_F1": round(val_th_res["rel_macro_strict"] * 100, 2),
            "Test_NER_Strict_Micro_F1": round(test_th_res["ner_micro_strict"] * 100, 2),
            "Test_NER_Relaxed_Micro_F1": round(test_th_res["ner_micro_relax"] * 100, 2),
            "Test_RE_Strict_Macro_F1": round(test_th_res["rel_macro_strict"] * 100, 2),
            "Test_RE_Relaxed_Macro_F1": round(test_th_res["rel_macro_relax"] * 100, 2),
            "Test_RE_Strict_Micro_F1": round(test_th_res["rel_micro_strict"] * 100, 2),
            "Test_RE_Relaxed_Micro_F1": round(test_th_res["rel_micro_relax"] * 100, 2)
        }
        append_to_csv(csv_out, row)

        if args.save_predictions:
            preds = test_th_res.pop("predictions", [])
            with open(output_dir / "per_class_report.json", "w", encoding="utf-8") as f:
                json.dump(test_th_res, f, indent=2, ensure_ascii=False)
            with open(output_dir / "test_predictions.json", "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2, ensure_ascii=False)
            logger.info(f"Relatório por classe e predições por frase salvos em: {output_dir.resolve()}")

    if args.no_save_ckpt and best_ckpt_dir.exists():
        shutil.rmtree(best_ckpt_dir)


if __name__ == "__main__":
    main()