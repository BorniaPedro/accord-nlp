# experiments/relex/augment_relex.py
import json
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_ENTITIES = {'object', 'property', 'quality', 'value'}
CANONICAL_RELATIONS = {
    'equal', 'greater', 'greater-equal', 'less', 'less-equal',
    'necessity', 'part-of', 'selection'
}


def tokenize_accord_style(text: str) -> list[str]:
    """Tokenizador que preserva a pontuação separada idêntica ao CODE-ACCORD."""
    text = re.sub(r'([,;:.!?()\[\]"\'/\\-])', r' \1 ', text)
    tokens = [t.strip() for t in text.split() if t.strip()]
    return tokens


def find_token_span(tokens: list[str], mention_tokens: list[str]) -> tuple[int, int] | None:
    """Localiza o span de tokens [start_idx, end_idx] (inclusivo)."""
    n_t = len(tokens)
    n_m = len(mention_tokens)
    if n_m == 0 or n_m > n_t:
        return None

    matches = []
    for i in range(n_t - n_m + 1):
        if tokens[i : i + n_m] == mention_tokens:
            matches.append((i, i + n_m - 1))

    # Retorna apenas se não houver ambiguidade (ocorrência única)
    return matches[0] if len(matches) == 1 else None


def compile_synthetic_batch(raw_samples: list[dict]) -> list[dict]:
    valid_documents = []
    rejected_count = 0

    for sample in raw_samples:
        sentence = sample.get("sentence", "").strip()
        entities = sample.get("entities", [])
        relations = sample.get("relations", [])

        if not sentence or not entities or not relations:
            rejected_count += 1
            continue

        tokens = tokenize_accord_style(sentence)

        # 1. Compila entidades
        ner_list = []
        entity_span_map = {}  # text -> index no ner_list
        valid_sample = True

        for ent in entities:
            mention = ent.get("text", "").strip()
            label = ent.get("label", "").strip()

            if label not in CANONICAL_ENTITIES:
                valid_sample = False
                break

            m_tokens = tokenize_accord_style(mention)
            span = find_token_span(tokens, m_tokens)

            if span is None:
                valid_sample = False
                break

            ent_idx = len(ner_list)
            ner_list.append([span[0], span[1], label])
            entity_span_map[mention] = ent_idx

        if not valid_sample:
            rejected_count += 1
            continue

        # 2. Compila relações
        re_list = []
        for rel in relations:
            h_text = rel.get("head", "").strip()
            t_text = rel.get("tail", "").strip()
            r_label = rel.get("relation", "").strip()

            if r_label not in CANONICAL_RELATIONS:
                valid_sample = False
                break

            if h_text not in entity_span_map or t_text not in entity_span_map:
                valid_sample = False
                break

            h_idx = entity_span_map[h_text]
            t_idx = entity_span_map[t_text]
            re_list.append([h_idx, t_idx, r_label])

        if not valid_sample or len(re_list) == 0:
            rejected_count += 1
            continue

        valid_documents.append({
            "tokenized_text": tokens,
            "ner": ner_list,
            "relations": re_list
        })

    logger.info(f"Compilação concluída: {len(valid_documents)} aceitos | {rejected_count} rejeitados por falha de alinhamento.")
    return valid_documents


def merge_with_train(
    synthetic_raw_file="data/relex/synthetic_raw.json",
    original_train_file="data/relex/relex_train.json",
    output_file="data/relex/relex_train_augmented.json"
):
    synth_path = Path(synthetic_raw_file)
    if not synth_path.exists():
        logger.error(f"Arquivo {synthetic_raw_file} não encontrado. Salve a saída do LLM primeiro.")
        return

    with open(synth_path, 'r', encoding='utf-8') as f:
        raw_synth = json.load(f)

    valid_synth = compile_synthetic_batch(raw_synth)

    with open(original_train_file, 'r', encoding='utf-8') as f:
        original_train = json.load(f)

    augmented_train = original_train + valid_synth

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_train, f, indent=2)

    logger.info(f"Dataset aumentado gerado: {len(original_train)} originais + {len(valid_synth)} sintéticos = {len(augmented_train)} total.")
    logger.info(f"Salvo em: {output_file}")


if __name__ == "__main__":
    merge_with_train()