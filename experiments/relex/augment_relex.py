import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_ENTITIES = {"object", "property", "quality", "value"}
CANONICAL_RELATIONS = {
    "equal", "greater", "greater-equal", "less", "less-equal",
    "necessity", "part-of", "selection"
}


def tokenize_accord_style(text: str) -> list[str]:
    """Tokenizador que preserva pontuações isoladas no padrão CODE-ACCORD."""
    text = re.sub(r'([,;:.!?()\[\]"\'/\\-])', r" \1 ", text)
    return [t.strip() for t in text.split() if t.strip()]


def find_token_span(tokens: list[str], mention_tokens: list[str]) -> tuple[int, int] | None:
    """Localiza o span de tokens [start_idx, end_idx] (inclusivo) quando único."""
    n_t, n_m = len(tokens), len(mention_tokens)
    if n_m == 0 or n_m > n_t:
        return None
    matches = [
        (i, i + n_m - 1)
        for i in range(n_t - n_m + 1)
        if tokens[i : i + n_m] == mention_tokens
    ]
    return matches[0] if len(matches) == 1 else None


def compile_synthetic_batch(raw_samples: list[dict]) -> list[dict]:
    """Valida offsets, alinha spans de tokens e filtra instâncias inconsistentes."""
    valid_documents = []
    rejected_count = 0

    for sample in raw_samples:
        sentence = (sample.get("sentence") or sample.get("text") or "").strip()
        entities = sample.get("entities", [])
        relations = sample.get("relations", [])

        if not sentence or not entities or not relations:
            rejected_count += 1
            continue

        tokens = tokenize_accord_style(sentence)
        if not tokens:
            rejected_count += 1
            continue

        tok_char_spans = []
        curr = 0
        for t in tokens:
            idx = sentence.find(t, curr)
            if idx != -1:
                tok_char_spans.append((idx, idx + len(t)))
                curr = idx + len(t)
            else:
                tok_char_spans.append((curr, curr + len(t)))
                curr += len(t) + 1

        ner_list = []
        entity_idx_by_offset = {}
        entity_idx_by_text = {}
        valid_sample = True

        for ent in entities:
            mention = ent.get("text", "").strip()
            label = ent.get("label", "").strip()

            if label not in CANONICAL_ENTITIES or not mention:
                valid_sample = False
                break

            span = None
            if "start" in ent and "end" in ent:
                s_char, e_char = ent["start"], ent["end"]
                s_tok, e_tok = None, None
                for t_i, (t_s, t_e) in enumerate(tok_char_spans):
                    if s_tok is None and t_s >= s_char:
                        s_tok = t_i
                    if t_e <= e_char:
                        e_tok = t_i
                if s_tok is not None and e_tok is not None and s_tok <= e_tok:
                    span = (s_tok, e_tok)

            if span is None:
                span = find_token_span(tokens, tokenize_accord_style(mention))

            if span is None:
                valid_sample = False
                break

            ent_idx = len(ner_list)
            ner_list.append([span[0], span[1], label])

            if "start" in ent and "end" in ent:
                entity_idx_by_offset[(ent["start"], ent["end"])] = ent_idx
            entity_idx_by_text[mention] = ent_idx

        if not valid_sample:
            rejected_count += 1
            continue

        re_list = []
        for rel in relations:
            head = rel.get("head")
            tail = rel.get("tail")
            r_label = (rel.get("label") or rel.get("relation") or "").strip()

            if r_label not in CANONICAL_RELATIONS:
                valid_sample = False
                break

            h_idx, t_idx = None, None
            if isinstance(head, int) and 0 <= head < len(ner_list):
                h_idx = head
            elif isinstance(head, dict):
                h_idx = entity_idx_by_offset.get((head.get("start"), head.get("end")))
                if h_idx is None:
                    h_idx = entity_idx_by_text.get(head.get("text", "").strip())
            elif isinstance(head, str):
                h_idx = entity_idx_by_text.get(head.strip())

            if isinstance(tail, int) and 0 <= tail < len(ner_list):
                t_idx = tail
            elif isinstance(tail, dict):
                t_idx = entity_idx_by_offset.get((tail.get("start"), tail.get("end")))
                if t_idx is None:
                    t_idx = entity_idx_by_text.get(tail.get("text", "").strip())
            elif isinstance(tail, str):
                t_idx = entity_idx_by_text.get(tail.strip())

            if h_idx is None or t_idx is None:
                valid_sample = False
                break

            re_list.append([h_idx, t_idx, r_label])

        if not valid_sample or not re_list:
            rejected_count += 1
            continue

        valid_documents.append({
            "tokenized_text": tokens,
            "ner": ner_list,
            "relations": re_list
        })

    logger.info(
        f"Validação concluída: {len(valid_documents)} aceitos | "
        f"{rejected_count} rejeitados por desalinhamento."
    )
    return valid_documents


def merge_with_train(synthetic_raw_file: str, original_train_file: str, output_file: str):
    synth_path = Path(synthetic_raw_file)
    if not synth_path.exists():
        logger.error(f"Arquivo sintético não encontrado: {synth_path}")
        return

    with open(synth_path, "r", encoding="utf-8") as f:
        raw_synth = json.load(f)

    valid_synth = compile_synthetic_batch(raw_synth)

    with open(original_train_file, "r", encoding="utf-8") as f:
        original_train = json.load(f)

    augmented_train = original_train + valid_synth
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(augmented_train, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Dataset consolidado: {len(original_train)} originais + "
        f"{len(valid_synth)} sintéticos = {len(augmented_train)} sentenças -> {output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida lotes sintéticos JSON e consolida com o treino.")
    parser.add_argument("--synthetic_raw", type=str, default="data/relex/synthetic_raw_600.json")
    parser.add_argument("--train_path", type=str, default="data/relex/relex_train.json")
    parser.add_argument("--output_path", type=str, default="data/relex/relex_train_augmented.json")
    args = parser.parse_args()
    merge_with_train(args.synthetic_raw, args.train_path, args.output_path)