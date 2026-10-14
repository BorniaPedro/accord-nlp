# experiments/relex/benchmark_timing.py
import json
import time
from pathlib import Path
import torch

from gliner import GLiNER

CANONICAL_ENTITIES = ['object', 'property', 'quality', 'value']
CANONICAL_RELATIONS = [
    'equal', 'greater', 'greater-equal', 'less', 'less-equal',
    'necessity', 'part-of', 'selection'
]


def run_benchmark(
    test_path="data/relex/relex_test.json",
    train_path="data/relex/relex_train.json",
    model_path="models/gliner_relex_large",
    num_samples=15,
    target_epochs=10
):
    print("=" * 70)
    print("PROFILING DE PERFORMANCE E TEMPO - GLINER-RELEX NATIVO")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware detectado      : {device.upper()}")
    if device == "cuda":
        print(f"Dispositivo GPU         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total              : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Aviso: Rodando em CPU. Na GPU/VM a velocidade será de 10x a 20x mais rápida.")
    print("-" * 70)

    # 1. Carrega modelo local
    print(f"-> Carregando modelo de: {model_path}...")
    t0_load = time.perf_counter()
    model = GLiNER.from_pretrained(model_path).to(device)
    t_load = time.perf_counter() - t0_load
    print(f"   Modelo carregado em {t_load:.2f} s")

    # 2. Carrega dados de teste e treino
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    total_test = len(test_data)
    total_train = len(train_data)
    print(f"Tamanho do Teste        : {total_test} sentenças")
    print(f"Tamanho do Treino       : {total_train} sentenças")

    # 3. Profiling de Inferência Conjunta (Entidades + Relações)
    print(f"\n-> Medindo inferência conjunta em {num_samples} sentenças do CODE-ACCORD...")
    sample_texts = [" ".join(doc['tokenized_text']) for doc in test_data[:num_samples]]

    # Passo de warmup
    _ = model.inference(
        texts=[sample_texts[0]],
        labels=CANONICAL_ENTITIES,
        relations=CANONICAL_RELATIONS,
        threshold=0.3,
        relation_threshold=0.5,
        return_relations=True,
        flat_ner=False
    )

    t0_inf = time.perf_counter()
    entities, relations = model.inference(
        texts=sample_texts,
        labels=CANONICAL_ENTITIES,
        relations=CANONICAL_RELATIONS,
        threshold=0.3,
        relation_threshold=0.5,
        return_relations=True,
        flat_ner=False
    )
    inf_elapsed = time.perf_counter() - t0_inf

    sec_per_sent = inf_elapsed / num_samples
    total_test_time = sec_per_sent * total_test

    print(f"   Tempo médio por sentença (inferência): {sec_per_sent:.3f} s")
    print(f"   Tempo total para avaliar Teste ({total_test} sentenças): {total_test_time:.1f} s ({total_test_time/60:.2f} min)")

    # Exemplo de extração
    print("\n   [Exemplo Real do Dataset (Amostra 0)]:")
    print(f"   Texto: \"{sample_texts[0][:80]}...\"")
    print(f"   Entidades detectadas : {len(entities[0])}")
    print(f"   Relações detectadas  : {len(relations[0])}")
    for r in relations[0][:4]:
        print(f"     -> {r['head']['text']} --[{r['relation']}]--> {r['tail']['text']} (score: {r['score']:.2f})")

    # 4. Projeção de Treinamento
    sec_per_train_step = sec_per_sent * 2.5
    sec_per_epoch = sec_per_train_step * total_train
    total_train_projected = sec_per_epoch * target_epochs

    print("\n" + "=" * 70)
    print("PROJEÇÃO CONSOLIDADA DE EXECUÇÃO")
    print("=" * 70)
    print(f"Avaliação do Test Set ({total_test} sentenças) : {total_test_time/60:.2f} minutos")
    print(f"Treino por Época ({total_train} sentenças)       : {sec_per_epoch/60:.2f} minutos")
    print(f"Treino Total ({target_epochs} épocas)              : {total_train_projected/60:.2f} minutos ({total_train_projected/3600:.2f} h)")
    print("=" * 70)


if __name__ == '__main__':
    run_benchmark()