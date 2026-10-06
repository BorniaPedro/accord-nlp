# Created by Hansi at 28/08/2023
# Updated for multi-backend NER (RoBERTa / GLiNER) and ontological filtering
import logging
import pandas as pd
import torch
from nltk import word_tokenize

from accord_nlp.information_extraction.convertor import entity_pairing, apply_ontological_filter, graph_building
from accord_nlp.text_classification.ner.ner_model import NERModel
from accord_nlp.text_classification.ner.gliner_model import GLiNERModel
from accord_nlp.text_classification.relation_extraction.re_model import REModel

logger = logging.getLogger(__name__)


class InformationExtractor:
    def __init__(
            self,
            ner_engine='gliner',
            ner_model_path=r"D:\Faculdade\TCC\resultados\gliner_model_best\checkpoint-390",
            ner_model_info=('roberta', 'ACCORD-NLP/ner-roberta-large'),
            re_model_info=('roberta', 'ACCORD-NLP/re-roberta-large'),
            gliner_threshold=0.50,
            apply_ontology=True,
            cuda_device=0,
            debug=False):
        """
        Initializes the end-to-end Information Extraction pipeline supporting multiple NER backends.

        :param ner_engine: str, 'gliner' (default/recommended) or 'roberta' (legacy baseline)
        :param ner_model_path: str, checkpoint path for GLiNER (used when ner_engine='gliner')
        :param ner_model_info: tuple, (<type>, <name>, <args>) (used when ner_engine='roberta')
        :param re_model_info: tuple, (<type>, <name>, <args>) for the Relation Extraction classifier
        :param gliner_threshold: float, confidence threshold for GLiNER span prediction
        :param apply_ontology: bool, whether to filter relation hallucinations using ontological rules
        :param cuda_device: int, GPU device index
        :param debug: bool, whether to enable detailed logging
        """
        self.debug = debug
        self.ner_engine = ner_engine.lower()
        self.apply_ontology = apply_ontology
        use_cuda = torch.cuda.is_available()

        if self.debug:
            logging.basicConfig(level=logging.INFO)

        # Initialize selected NER engine
        if self.ner_engine == 'gliner':
            logger.info("Initializing pipeline with GLiNER backend...")
            self.ner_model = GLiNERModel(
                model_path=ner_model_path,
                threshold=gliner_threshold,
                use_cuda=use_cuda,
                cuda_device=cuda_device
            )
        else:
            logger.info("Initializing pipeline with sequential RoBERTa backend...")
            ner_args = ner_model_info[2] if len(ner_model_info) > 2 else {}
            ner_args.update({'use_multiprocessing': False, 'use_multiprocessing_for_evaluation': False, 'process_count': 1})
            self.ner_model = NERModel(
                ner_model_info[0],
                ner_model_info[1],
                use_cuda=use_cuda,
                cuda_device=cuda_device,
                args=ner_args
            )

        # Initialize Relation Extraction engine
        re_args = re_model_info[2] if len(re_model_info) > 2 else {}
        re_args.update({'use_multiprocessing': False, 'use_multiprocessing_for_evaluation': False, 'process_count': 1})
        self.re_model = REModel(
            re_model_info[0],
            re_model_info[1],
            use_cuda=use_cuda,
            cuda_device=cuda_device,
            args=re_args
        )

    def preprocess(self, sentence):
        sentence = sentence.strip()
        sentence = ' '.join(word_tokenize(sentence))
        return sentence

    def extract_relations(self, sentence):
        """
        Runs the full end-to-end IE flow: Preprocess -> NER -> Pairing -> RE -> Ontological Filter.
        Returns a structured pd.DataFrame with predicted relations.
        """
        sentence = self.preprocess(sentence)

        # NER Inference
        try:
            ner_predictions, _ = self.ner_model.predict([sentence])
        except Exception as e:
            if self.debug:
                logger.warning(f"NER inference failed: {e}")
            return pd.DataFrame(columns=['output', 'prediction'])

        if not ner_predictions or len(ner_predictions[0]) == 0:
            return pd.DataFrame(columns=['output', 'prediction'])

        # Verify that at least two entity tokens exist
        detected_entities = [
            list(token_dict.values())[0]
            for token_dict in ner_predictions[0]
            if list(token_dict.values())[0] != 'O' and list(token_dict.values())[0].startswith('B-')
        ]
        if len(detected_entities) < 2:
            return pd.DataFrame(columns=['output', 'prediction'])

        # Candidate Entity Pairing
        try:
            entity_pair_df = entity_pairing(sentence, ner_predictions[0])
        except Exception as e:
            if self.debug:
                logger.warning(f"Entity pairing failed: {e}")
            return pd.DataFrame(columns=['output', 'prediction'])

        if entity_pair_df is None or len(entity_pair_df) == 0:
            return pd.DataFrame(columns=['output', 'prediction'])

        # Relation Classification
        try:
            re_predictions, _ = self.re_model.predict(entity_pair_df['output'].tolist())
            entity_pair_df['prediction'] = re_predictions
        except Exception as e:
            if self.debug:
                logger.warning(f"RE inference failed: {e}")
            return pd.DataFrame(columns=['output', 'prediction'])

        # Domain Ontology Filtering (if enabled)
        if self.apply_ontology:
            entity_pair_df = apply_ontological_filter(entity_pair_df)

        return entity_pair_df

    def sentence_to_graph(self, sentence, view=False):
        """
        Generates and optionally renders a dependency graph of the extracted relations.
        """
        entity_pair_df = self.extract_relations(sentence)
        if len(entity_pair_df) == 0:
            return None
        graph = graph_building(entity_pair_df, view=view)
        return graph