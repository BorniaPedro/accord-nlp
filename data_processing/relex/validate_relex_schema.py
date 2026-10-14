# data_processing/relex/validate_relex_schema.py
import argparse
import json
import os
from collections import Counter
from pathlib import Path


VALID_ENTITIES = {'object', 'property', 'quality', 'value'}
VALID_RELATIONS = {
    'equal', 'greater', 'greater-equal', 'less', 'less-equal',
    'necessity', 'part-of', 'selection', 'not-part-of'
}


def audit_split(file_path: Path):
    """
    Realiza a auditoria profunda de um arquivo JSON do ReLEx.
    Retorna (is_valid, num_docs, entity_counter, relation_counter, rel_docs_count, errors)
    """
    if not file_path.exists():
        return False, 0, Counter(), Counter(), 0, [f"Arquivo não encontrado: {file_path}"]

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return False, 0, Counter(), Counter(), 0, [f"Erro ao decodificar JSON: {e}"]

    if not isinstance(data, list):
        return False, 0, Counter(), Counter(), 0, ["O JSON raiz deve ser uma lista de documentos."]

    errors = []
    entity_counts = Counter()
    relation_counts = Counter()
    docs_with_relations = 0
    clean_sentences = set()

    for i, doc in enumerate(data):
        # 1. Checagem de Chaves
        for key in ('tokenized_text', 'ner', 'relations'):
            if key not in doc:
                errors.append(f"Doc {i}: Chave obrigatória '{key}' ausente.")

        tokens = doc.get('tokenized_text', [])
        ner = doc.get('ner', [])
        relations = doc.get('relations', [])

        # 2. Checagem de Tokens
        if not isinstance(tokens, list) or len(tokens) == 0:
            errors.append(f"Doc {i}: 'tokenized_text' deve ser uma lista não-vazia de strings.")
            continue

        clean_sentences.add(" ".join(tokens))

        # 3. Checagem de NER
        for n_idx, ent in enumerate(ner):
            if not isinstance(ent, list) or len(ent) != 3:
                errors.append(f"Doc {i}, Entidade {n_idx}: Formato inválido. Esperado [start, end, label].")
                continue

            s, e, l = ent
            if not (isinstance(s, int) and isinstance(e, int) and isinstance(l, str)):
                errors.append(f"Doc {i}, Entidade {n_idx}: Tipos incorretos em [start, end, label].")
                continue

            if s < 0 or e >= len(tokens) or s > e:
                errors.append(f"Doc {i}, Entidade {n_idx}: Índice fora dos limites (s={s}, e={e}, max={len(tokens)-1}).")

            if l not in VALID_ENTITIES:
                errors.append(f"Doc {i}, Entidade {n_idx}: Classe de entidade desconhecida: '{l}'.")

            entity_counts[l] += 1

        # 4. Checagem de Relações
        if len(relations) > 0:
            docs_with_relations += 1

        seen_pairs = set()
        for r_idx, rel in enumerate(relations):
            if not isinstance(rel, list) or len(rel) != 3:
                errors.append(f"Doc {i}, Relação {r_idx}: Formato inválido. Esperado [head_idx, tail_idx, label].")
                continue

            h, t, l = rel
            if not (isinstance(h, int) and isinstance(t, int) and isinstance(l, str)):
                errors.append(f"Doc {i}, Relação {r_idx}: Tipos incorretos em [head_idx, tail_idx, label].")
                continue

            if h < 0 or h >= len(ner) or t < 0 or t >= len(ner):
                errors.append(f"Doc {i}, Relação {r_idx}: Índice apontando fora de 'ner' (h={h}, t={t}, len_ner={len(ner)}).")

            if h == t:
                errors.append(f"Doc {i}, Relação {r_idx}: Auto-relação inválida (head == tail == {h}).")

            if l not in VALID_RELATIONS:
                errors.append(f"Doc {i}, Relação {r_idx}: Classe de relação desconhecida: '{l}'.")

            if (h, t, l) in seen_pairs:
                errors.append(f"Doc {i}, Relação {r_idx}: Relação duplicada detectada ({h}, {t}, '{l}').")
            seen_pairs.add((h, t, l))

            relation_counts[l] += 1

    is_valid = len(errors) == 0
    return is_valid, len(data), entity_counts, relation_counts, docs_with_relations, clean_sentences, errors


def main():
    parser = argparse.ArgumentParser(description="Validação rigorosa de Schema para Datasets GLiNER-ReLEx.")
    parser.add_argument("--data_dir", type=str, default="data/relex", help="Diretório onde estão os arquivos JSON.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits = ['relex_train.json', 'relex_val.json', 'relex_test.json']

    print("=" * 70)
    print("AUDITORIA E VALIDAÇÃO DE SCHEMA - GLINER-RELEX")
    print("=" * 70)

    all_passed = True
    split_sentences = {}

    for split_file in splits:
        file_path = data_dir / split_file
        is_valid, num_docs, ent_cnt, rel_cnt, rel_docs, sents, errors = audit_split(file_path)
        split_sentences[split_file] = sents

        status = "APROVADO [OK]" if is_valid else "FALHOU [ERRO]"
        print(f"\n-> Split: {split_file:<18} [{status}]")
        print(f"   Sentenças totais      : {num_docs}")
        print(f"   Com relações ativas   : {rel_docs} ({rel_docs/num_docs*100:.1f}%)" if num_docs else "")
        print(f"   Sem relações (hard neg): {num_docs - rel_docs} ({(num_docs - rel_docs)/num_docs*100:.1f}%)" if num_docs else "")
        print(f"   Total de Entidades    : {sum(ent_cnt.values())} {dict(ent_cnt)}")
        print(f"   Total de Relações     : {sum(rel_cnt.values())} {dict(rel_cnt)}")

        if not is_valid:
            all_passed = False
            print(f"   [!] Encontrados {len(errors)} erros no split. Primeiros 5:")
            for err in errors[:5]:
                print(f"       - {err}")

    # Checagem de Vazamento de Dados (Data Leakage)
    print("\n" + "-" * 70)
    print("TESTE DE VAZAMENTO DE DADOS (DATA LEAKAGE)")
    print("-" * 70)

    train_s = split_sentences.get('relex_train.json', set())
    val_s = split_sentences.get('relex_val.json', set())
    test_s = split_sentences.get('relex_test.json', set())

    leak_train_val = len(train_s & val_s)
    leak_train_test = len(train_s & test_s)
    leak_val_test = len(val_s & test_s)

    print(f"Interseção Treino <-> Validação : {leak_train_val} sentenças (Esperado: 0)")
    print(f"Interseção Treino <-> Teste      : {leak_train_test} sentenças (Esperado: 0)")
    print(f"Interseção Validação <-> Teste  : {leak_val_test} sentenças (Esperado: 0)")

    if leak_train_val > 0 or leak_train_test > 0 or leak_val_test > 0:
        all_passed = False
        print("[!] ATENÇÃO: Detectado vazamento de sentenças entre os splits!")
    else:
        print("[OK] Zero vazamento. Os splits são disjuntos.")

    print("=" * 70)
    if all_passed:
        print("RESULTADO FINAL: DATASET 100% VÁLIDO E PRONTO PARA TREINAMENTO.")
    else:
        print("RESULTADO FINAL: FALHAS ENCONTRADAS. CORRIJA OS ERROS ANTES DO TREINO.")
    print("=" * 70)


if __name__ == '__main__':
    main()