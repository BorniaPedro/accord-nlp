# experiments/ner/ner_experiments_gliner.py
import argparse
import os
import torch
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from gliner import GLiNER

from experiments.ner.evaluation import print_eval_ner
from experiments.ner.ner_config import SEED

TARGET_LABELS = ["object", "property", "quality", "value"]
HARD_NEGATIVES = ["component", "material", "action"]


def df_to_gliner_data(df, explicit_labels=True, include_hard_negatives=False):
    gliner_dataset = []
    
    for _, row in df.iterrows():
        tokens = row["processed_content"].split()
        tags = row["label"].split()
        
        spans = []
        start = None
        current_label = None
        
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if start is not None:
                    spans.append([start, i - 1, current_label])
                start = i
                current_label = tag[2:]
            elif tag.startswith("I-"):
                if start is not None and tag[2:] == current_label:
                    continue
                else:
                    if start is not None:
                        spans.append([start, i - 1, current_label])
                    start = i
                    current_label = tag[2:]
            else:  # 'O'
                if start is not None:
                    spans.append([start, i - 1, current_label])
                    start = None
                    current_label = None
                    
        if start is not None:
            spans.append([start, len(tags) - 1, current_label])
            
        entry = {
            "tokenized_text": tokens,
            "ner": spans
        }
        
        if explicit_labels:
            entry["ner_labels"] = TARGET_LABELS
            
        if include_hard_negatives:
            entry["ner_negatives"] = HARD_NEGATIVES
            
        gliner_dataset.append(entry)
        
    return gliner_dataset


def spans_to_bio_tags(tokens, predicted_entities, full_text):
    bio_tags = ["O"] * len(tokens)
    token_char_spans = []
    current_idx = 0
    
    for token in tokens:
        start_char = full_text.find(token, current_idx)
        if start_char == -1:
            start_char = current_idx
        end_char = start_char + len(token)
        token_char_spans.append((start_char, end_char))
        current_idx = end_char

    for entity in predicted_entities:
        ent_start = entity["start"]
        ent_end = entity["end"]
        label = entity["label"]

        matched_tokens = []
        for i, (tok_start, tok_end) in enumerate(token_char_spans):
            if max(ent_start, tok_start) < min(ent_end, tok_end):
                matched_tokens.append(i)

        if matched_tokens:
            bio_tags[matched_tokens[0]] = f"B-{label}"
            for i in matched_tokens[1:]:
                bio_tags[i] = f"I-{label}"
                
    return bio_tags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning e Avaliação GLiNER no CODE-ACCORD')
    
    # Modelo e Arquitetura
    parser.add_argument('--model_name', type=str, default="gliner-community/gliner_small-v2.5",
                        help='Nome ou path do modelo base')
    parser.add_argument('--max_width', type=int, default=12,
                        help='Tamanho maximo em tokens para spans candidatos')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Congela o text_encoder')

    # Treinamento e Otimizacao
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Numero total de epocas')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Tamanho do lote por dispositivo')
    parser.add_argument('--lr_encoder', type=float, default=1e-5,
                        help='Taxa de aprendizado do encoder')
    parser.add_argument('--lr_others', type=float, default=5e-5,
                        help='Taxa de aprendizado das camadas de span')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    
    # Loss e Negativos
    parser.add_argument('--focal_loss_alpha', type=float, default=-1.0,
                        help='Alpha da Focal Loss (-1 desativa)')
    parser.add_argument('--focal_loss_gamma', type=float, default=0.0,
                        help='Gamma da Focal Loss (0 desativa)')
    parser.add_argument('--negatives', type=float, default=1.0,
                        help='Taxa de amostragem negativa por batch')
    parser.add_argument('--masking', type=str, default="none", choices=["none", "label", "span", "global"],
                        help='Estrategia de mascaramento no treino')
    parser.add_argument('--use_hard_negatives', action='store_true',
                        help='Injeta classes negativas tecnicas')
    
    # Predicao e Saida
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Threshold de confianca para spans')
    parser.add_argument('--output_dir', type=str, default="outputs/gliner_ner",
                        help='Diretorio para salvar checkpoints e relatorios')
    
    args = parser.parse_args()

    # Checagem de acelerador
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executando em dispositivo: {device.upper()}")
    if device == "cuda":
        print(f"Placa detectada: {torch.cuda.get_device_name(0)}")

    train_df = Dataset.to_pandas(load_dataset('ACCORD-NLP/CODE-ACCORD-Entities', split='train'))
    test_df = Dataset.to_pandas(load_dataset('ACCORD-NLP/CODE-ACCORD-Entities', split='test'))

    train, eval_df = train_test_split(train_df, test_size=0.1, random_state=SEED)
    print(f"Dataset carregado | Treino: {train.shape[0]} | Validacao: {eval_df.shape[0]} | Teste: {test_df.shape[0]}")

    train_data = df_to_gliner_data(train, explicit_labels=True, include_hard_negatives=args.use_hard_negatives)
    eval_data = df_to_gliner_data(eval_df, explicit_labels=True, include_hard_negatives=False)

    print(f"\nCarregando checkpoint: {args.model_name}...")
    model = GLiNER.from_pretrained(args.model_name)
    model.config.max_width = args.max_width

    freeze_components = ["text_encoder"] if args.freeze_encoder else None

    print(f"\nIniciando treino ({args.num_epochs} epocas, batch={args.batch_size}, max_width={args.max_width}, gamma={args.focal_loss_gamma})...")
    model.train_model(
        train_dataset=train_data,
        eval_dataset=eval_data,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=-1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr_encoder,
        others_lr=args.lr_others,
        weight_decay=args.weight_decay,
        others_weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
        negatives=args.negatives,
        masking=args.masking,
        freeze_components=freeze_components,
    )

    print("\nExecutando predicao no conjunto de teste...")
    converted_predictions = []
    
    for _, row in test_df.iterrows():
        text = row["processed_content"]
        tokens = text.split()
        
        preds = model.predict_entities(text, TARGET_LABELS, threshold=args.threshold)
        bio_preds = spans_to_bio_tags(tokens, preds, text)
        converted_predictions.append(bio_preds)

    actuals = test_df["label"].tolist()
    actual_labels = [sub.split() for sub in actuals]

    os.makedirs(args.output_dir, exist_ok=True)
    eval_file_path = os.path.join(args.output_dir, 'test_eval_gliner.txt')
    
    print_eval_ner(actual_labels, converted_predictions, eval_file_path=eval_file_path)
    print(f"\nResultados salvos com sucesso em: {eval_file_path}")