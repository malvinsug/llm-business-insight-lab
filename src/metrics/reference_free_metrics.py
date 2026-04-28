"""
Please ensure all dependencies are installed. Also run:
`python -m spacy download en_core_web_sm`
"""

import logging
import re
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ReferenceFreeMetrics:
    """
    Lexical & Statistical Metrics
    Provides reference-free evaluation metrics for AI-generated text.
    """

    @staticmethod
    def _is_empty(text: Any) -> bool:
        """
        Check if the input text is valid (non-null and non-empty).

        Args:
            text (Any): The text to evaluate.

        Returns:
            bool: True if the text is missing or empty space, False otherwise.
        """
        if not isinstance(text, str):
            return True
        if not text.strip():
            return True
        return False

    @staticmethod
    def quantification_density(text: str) -> Dict[str, float]:
        """
        Compute density of quantities normalized by sentence count.
        Counts numerals, percentages, monetary values, temporal markers.

        Args:
            text (str): Input text containing business insight.

        Returns:
            Dict[str, float]: Quantification density score.
        """
        if ReferenceFreeMetrics._is_empty(text):
            return {"quantification_density": float("nan")}

        nlp = ModelCache.get_spacy()
        doc = nlp(text)

        num_sentences = max(len(list(doc.sents)), 1)
        count = 0

        target_labels = {"MONEY", "PERCENT", "DATE", "TIME", "QUANTITY", "CARDINAL"}
        for ent in doc.ents:
            if ent.label_ in target_labels:
                count += 1

        return {"quantification_density": count / num_sentences}

    @staticmethod
    def ngram_surprisal(text: str, model_name: str = "EleutherAI/pythia-160m") -> Dict[str, float]:
        """
        Compute mean and std token surprisal using a small causal LM from HuggingFace.

        Args:
            text (str): Input text.
            model_name (str): Identifier of the HuggingFace causal LM.

        Returns:
            Dict[str, float]: Mean and Standard Deviation of the log perplexity/surprisal.
        """
        if ReferenceFreeMetrics._is_empty(text):
            return {"mean_surprisal": float("nan"), "std_surprisal": float("nan")}

        import torch

        model, tokenizer = ModelCache.get_hf_model(model_name)

        device = model.device
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

        if inputs.input_ids.shape[1] < 2:
            return {"mean_surprisal": float("nan"), "std_surprisal": float("nan")}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            logits = outputs.logits

            # Compute token-level surprisal using cross entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            surprisal = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            surprisal_vals = surprisal.cpu().numpy()

        return {
            "mean_surprisal": float(surprisal_vals.mean()),
            "std_surprisal": float(surprisal_vals.std()),
        }

    @staticmethod
    def ner_and_domain_term_density(text: str) -> Dict[str, float]:
        """
        Extract unique ORG, PRODUCT, MONEY, PERCENT, DATE entities using Flair.
        Normalize by total word count.

        Args:
            text (str): Input text.

        Returns:
            Dict[str, float]: Dictionary comprising the normalized Unique NER density.
        """
        if ReferenceFreeMetrics._is_empty(text):
            return {"ner_density": float("nan")}

        from flair.data import Sentence

        tagger = ModelCache.get_flair_tagger()
        sentence = Sentence(text)
        tagger.predict(sentence)

        target_labels = {"ORG", "PRODUCT", "MONEY", "PERCENT", "DATE"}
        unique_entities = set()

        for entity in sentence.get_spans("ner"):
            if entity.tag in target_labels:
                unique_entities.add(entity.text.lower())

        word_count = max(len(text.split()), 1)

        return {"ner_density": len(unique_entities) / word_count}

    @staticmethod
    def causal_connective_density(text: str) -> Dict[str, float]:
        """
        Regex-based count of causal markers in text, normalized by word count.

        Args:
            text (str): Input text.

        Returns:
            Dict[str, float]: Density of causal markers.
        """
        if ReferenceFreeMetrics._is_empty(text):
            return {"causal_connective_density": float("nan")}

        connectives = [
            "because",
            "therefore",
            "as a result",
            "driven by",
            "given that",
            "leads to",
            "consequently",
            "thus",
        ]
        pattern = r"\b(?:" + "|".join(connectives) + r")\b"

        matches = re.finditer(pattern, text, re.IGNORECASE)
        count = sum(1 for _ in matches)
        word_count = max(len(text.split()), 1)

        return {"causal_connective_density": count / word_count}

    @staticmethod
    def evidentiality_index(text: str) -> Dict[str, float]:
        """
        Calculate the ratio of data-grounded evidentials vs ungrounded assertions.

        Args:
            text (str): Input text.

        Returns:
            Dict[str, float]: Evidentiality ratio and raw counts.
        """
        if ReferenceFreeMetrics._is_empty(text):
            return {
                "evidentiality_ratio": float("nan"),
                "grounded_count": float("nan"),
                "ungrounded_count": float("nan"),
            }

        grounded_terms = [
            "data shows",
            "kpis indicate",
            "metrics reveal",
            "shows",
            "indicates",
            "reveals",
            "based on",
            "demonstrates",
        ]
        ungrounded_terms = ["it is clear", "obviously", "undoubtedly", "clearly", "evident"]

        grounded_pattern = r"\b(?:" + "|".join(grounded_terms) + r")\b"
        ungrounded_pattern = r"\b(?:" + "|".join(ungrounded_terms) + r")\b"

        grounded_count = sum(1 for _ in re.finditer(grounded_pattern, text, re.IGNORECASE))
        ungrounded_count = sum(1 for _ in re.finditer(ungrounded_pattern, text, re.IGNORECASE))

        if ungrounded_count > 0:
            ratio = grounded_count / ungrounded_count
        elif grounded_count > 0:
            ratio = float(grounded_count)  # Assign a positive ratio if grounded elements found
        else:
            ratio = 0.0

        return {
            "evidentiality_ratio": float(ratio),
            "grounded_count": float(grounded_count),
            "ungrounded_count": float(ungrounded_count),
        }

    @staticmethod
    def hedging_calibration(text: str) -> Dict[str, float]:
        """
        Calculate the hedge-to-booster ratio and normalized frequencies.

        Args:
            text (str): Input text.

        Returns:
            Dict[str, float]: Ratio and semantic density.
        """
        if ReferenceFreeMetrics._is_empty(text):
            return {
                "hedge_to_booster_ratio": float("nan"),
                "hedge_density": float("nan"),
                "booster_density": float("nan"),
            }

        hedges = ["may", "could", "possibly", "suggest", "might", "perhaps", "likely", "probably"]
        boosters = ["definitely", "must", "always", "certainly", "absolutely"]

        hedge_pattern = r"\b(?:" + "|".join(hedges) + r")\b"
        booster_pattern = r"\b(?:" + "|".join(boosters) + r")\b"

        hedge_count = sum(1 for _ in re.finditer(hedge_pattern, text, re.IGNORECASE))
        booster_count = sum(1 for _ in re.finditer(booster_pattern, text, re.IGNORECASE))
        word_count = max(len(text.split()), 1)

        if booster_count > 0:
            ratio = hedge_count / booster_count
        elif hedge_count > 0:
            ratio = float(hedge_count)
        else:
            ratio = 0.0

        return {
            "hedge_to_booster_ratio": ratio,
            "hedge_density": hedge_count / word_count,
            "booster_density": booster_count / word_count,
        }

    @staticmethod
    def formality_score(text: str) -> Dict[str, float]:
        """
        Calculate algorithmic formality F-score depending on spaCy POS tagging distributions:
        F = (nouns + adjectives + prepositions) / (verbs + adverbs + pronouns + interjections).

        Args:
            text (str): Input text.

        Returns:
            Dict[str, float]: Formality score.
        """
        if ReferenceFreeMetrics._is_empty(text):
            return {"formality_score": float("nan")}

        nlp = ModelCache.get_spacy()
        doc = nlp(text)

        formal_tags = {"NOUN", "PROPN", "ADJ", "ADP"}
        informal_tags = {"VERB", "ADV", "PRON", "INTJ"}

        formal_count = 0
        informal_count = 0

        for token in doc:
            if token.pos_ in formal_tags:
                formal_count += 1
            elif token.pos_ in informal_tags:
                informal_count += 1

        if informal_count > 0:
            f_score = formal_count / informal_count
        elif formal_count > 0:
            f_score = float(formal_count)
        else:
            f_score = 0.0

        return {"formality_score": f_score}

    @staticmethod
    def imperative_to_declarative_ratio(text: str) -> Dict[str, float]:
        """
        Detect imperative vs declarative sentence mood using spaCy dependency parsing.

        Args:
            text (str): Input text subject to evaluation.

        Returns:
            Dict[str, float]: Ratio of imperative instructions vs statements.
        """
        if ReferenceFreeMetrics._is_empty(text):
            return {"imperative_to_declarative_ratio": float("nan")}

        nlp = ModelCache.get_spacy()
        doc = nlp(text)

        imperative_count = 0
        declarative_count = 0

        for sent in doc.sents:
            root = [token for token in sent if token.dep_ == "ROOT"]
            if not root:
                continue
            root_token = root[0]

            # Heuristic for detecting imperative clause
            if root_token.pos_ == "VERB":
                has_subj = any(
                    child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl")
                    for child in root_token.children
                )
                if not has_subj:
                    imperative_count += 1
                else:
                    declarative_count += 1
            else:
                declarative_count += 1

        if declarative_count > 0:
            ratio = imperative_count / declarative_count
        elif imperative_count > 0:
            ratio = float(imperative_count)
        else:
            ratio = 0.0

        return {"imperative_to_declarative_ratio": ratio}

    @staticmethod
    def modal_verb_profile(text: str) -> Dict[str, float]:
        """
        Frequency distribution of primary modal verbs.

        Args:
            text (str): Input text phrasing recommendations or findings.

        Returns:
            Dict[str, float]: Specific modal verb frequencies.
        """
        if ReferenceFreeMetrics._is_empty(text):
            return {
                "modal_should_freq": float("nan"),
                "modal_must_freq": float("nan"),
                "modal_could_freq": float("nan"),
                "modal_will_freq": float("nan"),
                "modal_may_freq": float("nan"),
                "modal_might_freq": float("nan"),
            }

        modals = ["should", "must", "could", "will", "may", "might"]
        word_count = max(len(text.split()), 1)

        # Tokenize by word-boundaries for robust matching
        words = re.findall(r"\b\w+\b", text.lower())

        results = {}
        for modal in modals:
            count = words.count(modal)
            results[f"modal_{modal}_freq"] = count / word_count

        return results


# Model Cache
class ModelCache:
    """
    Caches expensive NLP models like spaCy pipelines, flair taggers, and HuggingFace models
    as class attributes to avoid redundant loading across function calls.
    """

    _spacy_nlp = None
    _flair_tagger = None
    _hf_models: Dict[str, Any] = {}
    _hf_tokens: Dict[str, Any] = {}

    @classmethod
    def get_spacy(cls) -> Any:
        """Loads and caches the spaCy language model."""
        if cls._spacy_nlp is None:
            import spacy

            try:
                cls._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "Spacy model 'en_core_web_sm' not found. "
                    "Downloading it dynamically, but it is recommended to run: "
                    "python -m spacy download en_core_web_sm"
                )
                import spacy.cli

                spacy.cli.download("en_core_web_sm")
                cls._spacy_nlp = spacy.load("en_core_web_sm")
        return cls._spacy_nlp

    @classmethod
    def get_flair_tagger(cls) -> Any:
        """Loads and caches the flair NER tagger."""
        if cls._flair_tagger is None:
            from flair.models import SequenceTagger

            # ontonotes supports ORG, PRODUCT, MONEY, PERCENT, DATE, etc.
            cls._flair_tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
        return cls._flair_tagger

    @classmethod
    def get_hf_model(cls, model_name: str) -> tuple[Any, Any]:
        """Loads and caches HuggingFace transformers models/tokenizers."""
        if model_name not in cls._hf_models:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading HuggingFace model %s on %s...", model_name, device)
            cls._hf_tokens[model_name] = AutoTokenizer.from_pretrained(model_name)

            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            model.eval()
            cls._hf_models[model_name] = model

        return cls._hf_models[model_name], cls._hf_tokens[model_name]


if __name__ == "__main__":
    # Mock BACoT DataFrame
    mock_data = [
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "model": "kpi_analyser_newcot_pipeline",
            "department": "Sales",
            "human_answer": "Revenue grew by 15% in Q3 because sales increased. The team should continue this.",
            "answer": "The data shows a $15M revenue growth or 15% jump in Q3. This is evidently driven by increasing outbound sales in Q3 2024. Therefore, you should definitely invest more in this area because obviously it's highly profitable!",
        },
        {
            "id": "223e4567-e89b-12d3-a456-426614174001",
            "model": "kpi_analyser_nocot_pipeline",
            "department": "Development",
            "human_answer": "The product engagement is declining.",
            "answer": "KPIs indicate GitHub engagement may be declining relative to earlier trends. We might need to review these specific metrics further. Proceed with caution.",
        },
        {
            "id": "323e4567-e89b-12d3-a456-426614174002",
            "model": "kpi_analyser_newsymcot_pipeline",
            "department": "Customer Care",
            "human_answer": "",
            "answer": "",  # Should return NaN properly
        },
    ]

    df = pd.DataFrame(mock_data)

    logger.info("Initializing fast metrics evaluation script pipeline...")

    for idx, row in df.iterrows():
        text = row["answer"]
        logger.info("-" * 40)
        logger.info("Evaluating Example ID Context: %s", row.get("id", idx))
        logger.info("Text input: '%s'", text)

        res_quant = ReferenceFreeMetrics.quantification_density(text)
        logger.info("Quantification Density: %s", res_quant)

        # In a real environment without dependencies downloaded this may take time
        # Un-comment the next lines fully if models are present
        # res_surprisal = LexicalMetrics.ngram_surprisal(text, model_name="EleutherAI/pythia-160m")
        # logger.info("N-gram surprisal: %s", res_surprisal)

        # res_ner = LexicalMetrics.ner_and_domain_term_density(text)
        # logger.info("NER Density: %s", res_ner)

        res_causal = ReferenceFreeMetrics.causal_connective_density(text)
        logger.info("Causal connectives: %s", res_causal)

        res_evidentiality = ReferenceFreeMetrics.evidentiality_index(text)
        logger.info("Evidentiality: %s", res_evidentiality)

        res_hedging = ReferenceFreeMetrics.hedging_calibration(text)
        logger.info("Hedging Calibration: %s", res_hedging)

        # Requires spaCy
        # res_formality = LexicalMetrics.formality_score(text)
        # logger.info("Formality Score: %s", res_formality)

        # res_imperative = LexicalMetrics.imperative_to_declarative_ratio(text)
        # logger.info("Imperative Ratio: %s", res_imperative)

        res_modals = ReferenceFreeMetrics.modal_verb_profile(text)
        logger.info("Modal Profile: %s", res_modals)

    logger.info("Evaluation Mock Complete.")
