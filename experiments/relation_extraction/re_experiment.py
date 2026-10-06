# Created by Hansi at 21/07/2023
# Relation extraction train-test experiment
import argparse
import os
import shutil

import torch
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from accord_nlp.text_classification.relation_extraction.re_model import REModel
from experiments.relation_extraction.evaluation import macro_f1, macro_recall, macro_precision, print_eval_results, cls_report
from experiments.relation_extraction.re_config import re_args, SEED

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''evaluates multiple models   ''')
    parser.add_argument('--model_name', required=False, help='model name', default="bert-large-cased")
    parser.add_argument('--model_type', required=False, help='model type', default="bert")
    parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
    parser.add_argument('--wandb_api_key', required=False, help='wandb api key', default=None)
    parser.add_argument('--wandb_run_name', required=False, help='wandb run name', default=None)
    arguments = parser.parse_args()

    MODEL_TYPE = arguments.model_type
    MODEL_NAME = arguments.model_name
    cuda_device = int(arguments.cuda_device)
    re_args['wandb_project'] = 'relation-extraction'

    if arguments.wandb_api_key is not None:
        os.environ['WANDB_API_KEY'] = arguments.wandb_api_key

    if arguments.wandb_run_name is not None:
        re_args['wandb_kwargs'] = {'name': arguments.wandb_run_name}
    else:
        re_args['wandb_kwargs'] = {
            'name': f"{MODEL_NAME.split('/')[-1]}_{re_args['learning_rate']}_{re_args['num_train_epochs']}"}

    # Desativa concorrência interna do SimpleTransformers para compatibilidade com Windows
    re_args['use_multiprocessing'] = False
    re_args['use_multiprocessing_for_evaluation'] = False
    re_args['process_count'] = 1

    print("-> Carregando conjunto oficial de teste...")
    test_df = Dataset.to_pandas(load_dataset('ACCORD-NLP/CODE-ACCORD-Relations', split='test'))
    test_df = test_df.rename(columns={'tagged_sentence': 'text', 'relation_type': 'labels'})
    test_df = test_df[['example_id', 'text', 'labels']]

    # Carrega o modelo oficial direto do Hugging Face para teste (sem re-treinar)
    print(f"-> Inicializando modelo {MODEL_NAME}...")
    model = REModel(MODEL_TYPE, MODEL_NAME, use_cuda=torch.cuda.is_available(), cuda_device=cuda_device, args=re_args)

    print("-> Executando inferência sobre o conjunto canônico de teste...")
    predictions, raw_outputs = model.predict(test_df["text"].tolist())
    test_df['predictions'] = predictions

    os.makedirs(re_args['best_model_dir'], exist_ok=True)
    print_eval_results(test_df['labels'].tolist(), predictions,
                       eval_file_path=os.path.join(re_args['best_model_dir'], 'test_eval.txt'))
    test_df.to_csv(os.path.join(re_args['best_model_dir'], 'predictions.csv'), encoding='utf-8', index=False)
    print("-> Avaliação concluída com sucesso!")