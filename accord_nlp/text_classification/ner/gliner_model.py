# Created for GLiNER integration into accord_nlp
import logging
import re
import torch
from gliner import GLiNER

logger = logging.getLogger(__name__)


class GLiNERModel:
    """
    Wrapper for GLiNER models, compatible with the inference interface
    used by the ACCORD-NLP Information Extraction pipeline.
    """
    def __init__(
            self,
            model_path,
            labels=None,
            threshold=0.50,
            use_cuda=True,
            cuda_device=0,
            args=None):
        """
        :param model_path: str, directory path or Hugging Face hub identifier for the GLiNER checkpoint
        :param labels: list of str, entity types to extract (default: ['object', 'property', 'quality', 'value'])
        :param threshold: float, confidence threshold for entity span prediction (default: 0.50)
        :param use_cuda: bool, whether to use GPU acceleration
        :param cuda_device: int, CUDA device index
        """
        self.model_path = model_path
        self.labels = labels if labels is not None else ["object", "property", "quality", "value"]
        self.threshold = threshold
        self.device = f"cuda:{cuda_device}" if (use_cuda and torch.cuda.is_available()) else "cpu"

        logger.info(f"Loading GLiNER from '{model_path}' on device '{self.device}'...")
        self.model = GLiNER.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, sentences):
        """
        Generates token-level predictions following the BIO scheme for a list of sentences.

        :param sentences: list of str
        :return: (predictions, raw_outputs)
            predictions: list of sentences, where each sentence is a list of {token: 'B-tag' | 'I-tag' | 'O'}
            raw_outputs: list of raw entity spans predicted by GLiNER
        """
        all_predictions = []
        all_raw_outputs = []

        for sentence in sentences:
            sentence_clean = sentence.strip()
            # GLiNER span prediction
            spans = self.model.predict_entities(sentence_clean, labels=self.labels, threshold=self.threshold)
            all_raw_outputs.append(spans)

            # Deterministic character-boundary mapping (\S+ to prevent multi-space drift)
            tokens_spans = []
            for match in re.finditer(r'\S+', sentence_clean):
                tokens_spans.append((match.group(), match.start(), match.end()))

            sentence_token_preds = []
            seen_entities = set()

            for tok, t_start, t_end in tokens_spans:
                matched_tag = 'O'
                for ent_idx, ent in enumerate(spans):
                    # Check character overlap between token and predicted entity span
                    if not (t_end <= ent['start'] or t_start >= ent['end']):
                        if ent_idx not in seen_entities:
                            seen_entities.add(ent_idx)
                            matched_tag = f"B-{ent['label']}"
                        else:
                            matched_tag = f"I-{ent['label']}"
                        break

                sentence_token_preds.append({tok: matched_tag})

            all_predictions.append(sentence_token_preds)

        return all_predictions, all_raw_outputs