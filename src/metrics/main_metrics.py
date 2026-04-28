from bert_score import score as BERTScore
from textstat import flesch_kincaid_grade, flesch_reading_ease
from typing import Dict

from src.nli.minicheck import NLIMiniCheckEvaluator
from src.reasoning.evaluator import ReasoningEvaluator

from typing import Iterable

class MainMetrics:
    """
    Main Metrics based on master's thesis
    The goal of master's thesis is to measure the business insight text based on:
    - Clarity
    - Relevance
    - Factuality
    - Actionability
    - Reasoning
    """

    @staticmethod
    def evaluate_flesch_kincaid_grade(text:str) -> float:
        """
        Measures the readability formula that estimates the U.S. school grade level. (lower is better) It is based on average sentence length and average syllables per word. 

        Args:
            text (str): The text to evaluate.
        
        Returns:
            float: Flesch Kincaid Grade score.
        """
        return flesch_kincaid_grade(text)
    
    @staticmethod
    def evaluate_flesch_reading_ease(text:str) -> float:
        """
        Measures how easy a text is to read based on average sentence length and average syllables per word. 

        Args:
            text (str): The text to evaluate.
        
        Returns:
            float: Flesch Reading Ease score.
        """
        return flesch_reading_ease(text)
    
    @staticmethod
    def evaluate_bert_score(candidate:Iterable[str], reference:Iterable[str]) -> Dict[str,Iterable]:
        """
        Calculate the precision, recall and F1-Score of BERT score to understand the relevance of candidate towards the reference.

        Args:
            candidate(str): 
            reference(str)
        """
        P,R, F1 = BERTScore(candidate.to_list(),reference.to_list(), lang='en')

        return {
            "bert_precision": P,
            "bert_recall": R,
            "bert_f1": F1
        }
    
    @staticmethod
    def evaluate_factuality(facts_to_check:list[str], fact_reference:str) ->float:
        """
        Evaluate the precision of factuality given by AI-generated business insight text. 

        Args:
            facts_to_check(list[str]): observations and interpretations written by AI
            fact_reference(str): Reference to check the correctness of facts given by AI. It consists of observations and hypotheses made by human-written text.

        Returns:
            float: The precision score of the factuality of AI-generated business insight text.
        """
        from transformers.utils.logging import disable_progress_bar
        disable_progress_bar()

        nli_model = NLIMiniCheckEvaluator.get_nli_model()
        fact_check_dict: dict = NLIMiniCheckEvaluator.evaluate_answer_minicheck(
            nli_model = nli_model,
            claims_to_check=facts_to_check,
            context_document= fact_reference,
            k=5, 
            recommendation_or_fact_str="facts"
        )
        del nli_model

        factuality_precision = NLIMiniCheckEvaluator.calculate_factuality_precision(fact_check_dict=fact_check_dict, prefix="facts_50")

        return factuality_precision
    
    @staticmethod
    def evaluate_actionability(recommendations_to_check:list[str], recommendations_reference:str) ->float:
        """
        Evaluate the precision of actionability given by AI-generated business insight text. 

        Args:
            recommeendations_to_check(list[str]): business recommendations written by AI
            recommendations_reference(str): Reference to compare the actionability given by AI. It contains recommendations made by human-written text.

        Returns:
            float: The precision score of the actionability of AI-generated business insight text.
        """
        from transformers.utils.logging import disable_progress_bar
        disable_progress_bar()
        
        nli_model = NLIMiniCheckEvaluator.get_nli_model()
        rec_check_dict: dict = NLIMiniCheckEvaluator.evaluate_answer_minicheck(
            nli_model=nli_model,
            claims_to_check=recommendations_to_check,
            context_document= recommendations_reference,
            k=5, 
            recommendation_or_fact_str="recommendations"
        )
        del nli_model

        actionability_precision = NLIMiniCheckEvaluator.calculate_factuality_precision(fact_check_dict=rec_check_dict, prefix="recommendations_50")
        return actionability_precision
    
    @staticmethod
    def evaluate_reasoning(reasoning_dict_list:list|dict) -> dict:
        """
        Evaluate how well the reasoning of the AI-generated business insight text based on LLM.

        Args:
            reasoning_dict_list(list|dict): It could be a list of dictionary or a dictionary.
        Returns:
            dict[str,float|int]: total correct reasoning, total reasoning and correct reasoning percentage
        """
        
        correct_reasoning = ReasoningEvaluator.extract_reasoning_count(reasoning_dict_list)
        total_reasoning_to_verify = ReasoningEvaluator.extract_total_reasoning_to_verify(reasoning_dict_list)
        reasoning_percentage = correct_reasoning / total_reasoning_to_verify
        return {
            'correct_reasoning_count' : correct_reasoning,
            'total_reasoning_to_verify': total_reasoning_to_verify,
            'reasoning_percentage': reasoning_percentage
        }
