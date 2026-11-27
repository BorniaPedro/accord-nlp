# experiments/ie_gliner.py
import os
import re
import nltk
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from gliner import GLiNER
from nltk import word_tokenize

from accord_nlp.information_extraction.convertor import entity_pairing
from accord_nlp.text_classification.relation_extraction.re_model import REModel

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def extract_entities_from_tagged(text):
    m1 = re.search(r'<e1>(.*?)</e1>', text)
    m2 = re.search(r'<e2>(.*?)</e2>', text)
    e1 = m1.group(1).strip() if m1 else ""
    e2 = m2.group(1).strip() if m2 else ""
    return e1, e2


def clean_sentence(text):
    return re.sub(r'</?e[12]>', '', text).strip()


class GLiNERInformationExtractor:
    def __init__(self, gliner_path, re_model_name='ACCORD-NLP/re-roberta-large', cuda_device=0, threshold=0.50):
        print(f"-> Carregando GLiNER de: {gliner_path}")
        self.gliner_model = GLiNER.from_pretrained(gliner_path)
        if torch.cuda.is_available():
            self.gliner_model = self.gliner_model.to(f'cuda:{cuda_device}')
        
        self.labels = ["object", "property", "quality", "value"]
        self.threshold = threshold

        print(f"-> Carregando RE oficial: {re_model_name}")
        re_args = {
            'use_multiprocessing': False,
            'use_multiprocessing_for_evaluation': False,
            'process_count': 1
        }
        self.re_model = REModel('roberta', re_model_name, use_cuda=torch.cuda.is_available(),
                                cuda_device=cuda_device, args=re_args)

    def preprocess(self, sentence):
        sentence = sentence.strip()
        return ' '.join(word_tokenize(sentence))

    def predict_ner(self, sentence):
        """
        Alinhamento de spans com tokens via limites de caracteres (re.finditer).
        Garante marcação BIO exata sem drift de índice por espaços.
        """
        entities = self.gliner_model.predict_entities(sentence, labels=self.labels, threshold=self.threshold)
        
        # Mapeia tokens reais (ignora espaços múltiplos) com coordenadas exatas
        tokens_spans = []
        for match in re.finditer(r'\S+', sentence):
            tokens_spans.append((match.group(), match.start(), match.end()))

        token_preds = []
        seen_entities = set()

        for tok, t_start, t_end in tokens_spans:
            matched_tag = 'O'
            for ent_idx, ent in enumerate(entities):
                # Verifica sobreposição entre o token e o span da entidade
                if not (t_end <= ent['start'] or t_start >= ent['end']):
                    if ent_idx not in seen_entities:
                        seen_entities.add(ent_idx)
                        matched_tag = f"B-{ent['label']}"
                    else:
                        matched_tag = f"I-{ent['label']}"
                    break
            token_preds.append({tok: matched_tag})

        return [token_preds]

    def apply_ontological_filter(self, entity_pair_df):
        """
        Descarta alucinações do RE incompatíveis com a ontologia das normas.
        """
        comp_relations = {'equal', 'greater', 'greater-equal', 'less', 'less-equal'}
        filtered_preds = []

        for _, row in entity_pair_df.iterrows():
            pred = row['prediction']
            if pred == 'none':
                filtered_preds.append('none')
                continue

            # As tags salvas pelo convertor.py vêm limpas (ex: 'object', 'value')
            t1 = str(row['<e1>_tag']).lower().replace('b-', '').replace('i-', '')
            t2 = str(row['<e2>_tag']).lower().replace('b-', '').replace('i-', '')

            # 1. Dois valores numéricos nunca formam relação normativa entre si
            if t1 == 'value' and t2 == 'value':
                filtered_preds.append('none')
                continue

            # 2. Relações de comparação numérica exigem que ao menos um membro seja 'value'
            if pred in comp_relations and (t1 != 'value' and t2 != 'value'):
                filtered_preds.append('none')
                continue

            # 3. Relação 'part-of' é física/estrutural e nunca envolve números
            if pred == 'part-of' and (t1 == 'value' or t2 == 'value'):
                filtered_preds.append('none')
                continue

            filtered_preds.append(pred)

        entity_pair_df['prediction'] = filtered_preds
        return entity_pair_df

    def extract_relations(self, sentence):
        sentence = self.preprocess(sentence)
        
        # 1. Inferência com GLiNER
        try:
            ner_predictions = self.predict_ner(sentence)
        except Exception:
            return pd.DataFrame(columns=['output', 'prediction'])

        # Verifica se há pelo menos duas entidades detectadas
        entidades_detectadas = [
            list(d.values())[0] for d in ner_predictions[0] 
            if list(d.values())[0].startswith('B-')
        ]
        if len(entidades_detectadas) < 2:
            return pd.DataFrame(columns=['output', 'prediction'])

        # 2. Pareamento com convertor.py
        try:
            entity_pair_df = entity_pairing(sentence, ner_predictions[0])
        except Exception:
            return pd.DataFrame(columns=['output', 'prediction'])

        if entity_pair_df is None or len(entity_pair_df) == 0:
            return pd.DataFrame(columns=['output', 'prediction'])

        # 3. RE com RoBERTa oficial
        try:
            re_predictions, _ = self.re_model.predict(entity_pair_df['output'].tolist())
            entity_pair_df['prediction'] = re_predictions
        except Exception:
            return pd.DataFrame(columns=['output', 'prediction'])

        # 4. Aplicação do Filtro Ontológico
        entity_pair_df = self.apply_ontological_filter(entity_pair_df)

        return entity_pair_df


if __name__ == '__main__':
    print("=" * 60)
    print("BENCHMARK FIM A FIM: GLiNER + RE (COM FILTRO ONTOLÓGICO)")
    print("=" * 60)

    GLINER_MODEL_PATH = r"D:\Faculdade\TCC\resultados\gliner_model_best\checkpoint-390"

    print("-> Carregando partição canônica de teste...")
    test_ds = load_dataset('ACCORD-NLP/CODE-ACCORD-Relations', split='test')
    test_df = Dataset.to_pandas(test_ds)

    grouped_sentences = {}
    for _, row in test_df.iterrows():
        tagged = row['tagged_sentence']
        rel = row['relation_type']
        s_clean = clean_sentence(tagged)

        if s_clean not in grouped_sentences:
            grouped_sentences[s_clean] = {'sentence': s_clean, 'gold_triplets': set()}

        if rel != 'none':
            e1, e2 = extract_entities_from_tagged(tagged)
            grouped_sentences[s_clean]['gold_triplets'].add((e1.lower(), rel, e2.lower()))

    total_gold_relations = sum(len(d['gold_triplets']) for d in grouped_sentences.values())
    print(f"-> Total de relações ativas no gabarito: {total_gold_relations}")

    ie = GLiNERInformationExtractor(gliner_path=GLINER_MODEL_PATH, cuda_device=0, threshold=0.50)

    classes_list = ['equal', 'greater', 'greater-equal', 'less', 'less-equal', 'necessity', 'part-of', 'selection']
    tp_by_class = {c: 0 for c in classes_list}
    fp_by_class = {c: 0 for c in classes_list}
    fn_by_class = {c: 0 for c in classes_list}

    print("-> Processando inferência com GLiNER alinhado e filtrado...")
    for s_clean, data in tqdm(grouped_sentences.items(), desc="Avaliando IE (Otimizado)"):
        sentence = data['sentence']
        gold_triplets = data['gold_triplets']

        pred_df = ie.extract_relations(sentence)

        pred_triplets = set()
        if len(pred_df) > 0:
            for _, row in pred_df.iterrows():
                pred_rel = row['prediction']
                if pred_rel != 'none':
                    p_e1, p_e2 = extract_entities_from_tagged(row['output'])
                    pred_triplets.add((p_e1.lower(), pred_rel, p_e2.lower()))

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

    total_tp = sum(tp_by_class.values())
    total_fp = sum(fp_by_class.values())
    total_fn = sum(fn_by_class.values())

    print("\n" + "=" * 65)
    print("RELATÓRIO GLiNER + RE ACCORD (ALINHADO + FILTRADO)")
    print("=" * 65)
    print(f"{'Classe de Relação':<16} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Suporte':<8}")
    print("-" * 65)

    precisions, recalls, f1s = [], [], []
    for c in classes_list:
        tp, fp, fn = tp_by_class[c], fp_by_class[c], fn_by_class[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        if (tp + fn) > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        print(f"{c:<16} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {tp+fn:<8}")

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

    os.makedirs("outputs", exist_ok=True)
    out_file = os.path.join("outputs", "ie_gliner_filtered_test_eval.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Micro F1: {micro_f1:.4f}\nMacro F1: {macro_f1:.4f}\nMacro Precision: {macro_p:.4f}\nMacro Recall: {macro_r:.4f}\n")
    print(f"Relatório salvo em: {out_file}")