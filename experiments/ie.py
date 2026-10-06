# Created by Hansi on 26/03/2024
# Updated for unified End-to-End Information Extraction benchmark evaluation
import argparse
import os
import re
import nltk
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

from accord_nlp.information_extraction.ie_pipeline import InformationExtractor

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def extract_entities_from_tagged(text):
    """Extracts entity texts enclosed within <e1> and <e2> tags."""
    m1 = re.search(r'<e1>(.*?)</e1>', text)
    m2 = re.search(r'<e2>(.*?)</e2>', text)
    e1 = m1.group(1).strip() if m1 else ""
    e2 = m2.group(1).strip() if m2 else ""
    return e1, e2


def clean_sentence(text):
    """Strips markup tags to restore the original raw text sentence."""
    return re.sub(r'</?e[12]>', '', text).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Information Extraction Benchmark Evaluation")
    parser.add_argument(
        '--engine',
        type=str,
        default='gliner',
        choices=['gliner', 'roberta'],
        help="NER engine to use ('gliner' or 'roberta'). Default: 'gliner'"
    )
    parser.add_argument(
        '--gliner_path',
        type=str,
        default="accord_nlp/models/gliner_large_best",
        help="Path to trained GLiNER checkpoint directory."
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.50,
        help="Confidence threshold for GLiNER span prediction. Default: 0.50"
    )
    parser.add_argument(
        '--no_ontology',
        action='store_true',
        help="Disable domain ontological filtering (leave raw RE predictions)."
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help="Optional custom output filename. If None, generated automatically based on configuration."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    apply_ontology = not args.no_ontology

    # Define automatic report output name
    if args.output_file:
        out_filename = args.output_file
    else:
        if args.engine == 'gliner':
            suffix = "filtered" if apply_ontology else "raw"
            out_filename = f"ie_gliner_{suffix}_test_eval.txt"
        else:
            out_filename = "ie_baseline_test_eval.txt"

    print("=" * 65)
    print("END-TO-END INFORMATION EXTRACTION BENCHMARK EVALUATION")
    print("=" * 65)
    print(f"NER Engine          : {args.engine.upper()}")
    if args.engine == 'gliner':
        print(f"GLiNER Checkpoint   : {args.gliner_path}")
        print(f"Confidence Threshold: {args.threshold}")
    print(f"Ontological Filter  : {'ENABLED' if apply_ontology else 'DISABLED'}")
    print(f"Output File Destination : outputs/{out_filename}")
    print("=" * 65)

    # Load canonical test split
    print("-> Loading canonical test dataset (ACCORD-NLP/CODE-ACCORD-Relations)...")
    test_ds = load_dataset('ACCORD-NLP/CODE-ACCORD-Relations', split='test')
    test_df = Dataset.to_pandas(test_ds)

    # Group by clean sentence to evaluate all triplets jointly per sentence
    grouped_sentences = {}
    for _, row in test_df.iterrows():
        tagged = row['tagged_sentence']
        rel = row['relation_type']
        s_clean = clean_sentence(tagged)

        if s_clean not in grouped_sentences:
            grouped_sentences[s_clean] = {
                'sentence': s_clean,
                'gold_triplets': set(),
                'gold_pairs_count': 0
            }

        grouped_sentences[s_clean]['gold_pairs_count'] += 1
        if rel != 'none':
            e1, e2 = extract_entities_from_tagged(tagged)
            grouped_sentences[s_clean]['gold_triplets'].add((e1.lower(), rel, e2.lower()))

    print(f"-> Total unique sentences: {len(grouped_sentences)}")
    total_gold_relations = sum(len(d['gold_triplets']) for d in grouped_sentences.values())
    print(f"-> Total active normative triplets in ground truth: {total_gold_relations}")

    # Instantiate InformationExtractor via integrated core package
    print("-> Initializing InformationExtractor pipeline...")
    ie = InformationExtractor(
        ner_engine=args.engine,
        ner_model_path=args.gliner_path,
        gliner_threshold=args.threshold,
        apply_ontology=apply_ontology,
        cuda_device=0,
        debug=False
    )

    # End-to-end evaluation loop
    classes_list = ['equal', 'greater', 'greater-equal', 'less', 'less-equal', 'necessity', 'part-of', 'selection']
    tp_by_class = {c: 0 for c in classes_list}
    fp_by_class = {c: 0 for c in classes_list}
    fn_by_class = {c: 0 for c in classes_list}

    print("-> Running end-to-end extraction across test sentences...")
    for s_clean, data in tqdm(grouped_sentences.items(), desc=f"Evaluating IE ({args.engine.upper()})"):
        sentence = data['sentence']
        gold_triplets = data['gold_triplets']

        # Delegate execution directly to integrated library method
        pred_df = ie.extract_relations(sentence)

        pred_triplets = set()
        if len(pred_df) > 0:
            for _, row in pred_df.iterrows():
                pred_rel = row['prediction']
                if pred_rel != 'none':
                    p_e1, p_e2 = extract_entities_from_tagged(row['output'])
                    pred_triplets.add((p_e1.lower(), pred_rel, p_e2.lower()))

        # Strict Triplet Match logic
        matched_gold = set()
        for p_trip in pred_triplets:
            p_e1, p_rel, p_e2 = p_trip
            if p_trip in gold_triplets:
                if p_rel in tp_by_class:
                    tp_by_class[p_rel] += 1
                matched_gold.add(p_trip)
            else:
                if p_rel in fp_by_class:
                    fp_by_class[p_rel] += 1

        for g_trip in gold_triplets:
            if g_trip not in matched_gold:
                g_rel = g_trip[1]
                if g_rel in fn_by_class:
                    fn_by_class[g_rel] += 1

    # 5. Metrics calculation and consolidation
    total_tp = sum(tp_by_class.values())
    total_fp = sum(fp_by_class.values())
    total_fn = sum(fn_by_class.values())

    print("\n" + "=" * 65)
    print("END-TO-END INFORMATION EXTRACTION REPORT (STRICT MATCH)")
    print("=" * 65)
    print(f"{'Relation Class':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)

    precisions, recalls, f1s = [], [], []

    for c in classes_list:
        tp = tp_by_class[c]
        fp = fp_by_class[c]
        fn = fn_by_class[c]
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        print(f"{c:<16} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {support:<8}")

    macro_p = sum(precisions) / len(precisions) if precisions else 0.0
    macro_r = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    print("-" * 65)
    print(f"{'Micro Avg':<16} | {micro_p:<10.4f} | {micro_r:<10.4f} | {micro_f1:<10.4f} | {total_gold_relations:<8}")
    print(f"{'Macro Avg':<16} | {macro_p:<10.4f} | {macro_r:<10.4f} | {macro_f1:<10.4f} | {total_gold_relations:<8}")
    print("=" * 65)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, out_filename)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"NER Engine: {args.engine}\n")
        f.write(f"Ontology Filter: {apply_ontology}\n")
        f.write(f"Micro F1: {micro_f1:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Macro Precision: {macro_p:.4f}\n")
        f.write(f"Macro Recall: {macro_r:.4f}\n")
    print(f"\n-> Report successfully saved to: {out_file}")