import os
import openai

from src.utils.config import AZURE_DEPLOYMENT, AZURE_BASE_URL, AZURE_API_VERSION, AZURE_API_KEY, OPENAI_API_KEY, JUDGE_SYSTEM_PROMPT_FILE_DIR, JUDGE_TEMPERATURE

class ReasoningEvaluator:
    """
    A reasoning evaluator using LLM (LLM-as-a-Judge) for business insight text.
    """


    AZURE_DEPLOYMENT = AZURE_DEPLOYMENT
    AZURE_BASE_URL = AZURE_BASE_URL
    AZURE_API_VERSION = AZURE_API_VERSION
    AZURE_API_KEY = AZURE_API_KEY

    OPENAI_API_KEY = OPENAI_API_KEY

    @staticmethod
    def get_llm():
        try:
            llm: openai.AzureOpenAI = openai.AzureOpenAI(
                azure_endpoint=ReasoningEvaluator.AZURE_BASE_URL, 
                azure_deployment = ReasoningEvaluator.AZURE_DEPLOYMENT, 
                api_version = ReasoningEvaluator.AZURE_API_VERSION , 
                api_key = ReasoningEvaluator.AZURE_API_KEY
            )
        except:
            llm: openai.OpenAI = openai.OpenAI(
                api_key=ReasoningEvaluator.OPENAI_API_KEY
            )

        return llm



    JUDGE_SYSTEM_PROMPT_FILE_DIR = JUDGE_SYSTEM_PROMPT_FILE_DIR
    JUDGE_TEMPERATURE = JUDGE_TEMPERATURE

    @staticmethod
    def get_system_prompt():
        with open(ReasoningEvaluator.JUDGE_SYSTEM_PROMPT_FILE_DIR) as f:
            system_prompt = f.read()
        
        return system_prompt
    
    @staticmethod
    def extract_reasoning_json_from_llm(data:str) -> dict:
            messages = [
                {"role": "system", "content": ReasoningEvaluator.get_system_prompt()},
                {"role": "user", "content": data}
            ]
            response = ReasoningEvaluator.get_llm().chat.completions.create(
                model=ReasoningEvaluator.AZURE_DEPLOYMENT,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
    
    def wrap_data_as_user_message(question:str, answer: str, human_observation:str) -> str:
        return f"<LLM generated text>\n{answer}\n<\LLM generated text>\n\n<Human Observations and Hypotheses>{human_observation}\n<\Human Observations and Hypotheses>"


    @staticmethod
    def evaluate_reasoning(question:str,answer:str, human_observation:str):
        user_message = ReasoningEvaluator.wrap_data_as_user_message(question,answer, human_observation)
        return ReasoningEvaluator.extract_reasoning_json_from_llm(user_message)
