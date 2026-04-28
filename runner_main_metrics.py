import pandas as pd
from src.metrics.main_metrics import MainMetrics
import logging
import pickle

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import transformers

transformers.utils.logging.set_verbosity_error()


def build_bacot_data_from_pickle():
    logger.info("Fetching BACoT basic info dataframe...")
    with open("data/bacot_basic_info.pkl", "rb") as f:
        basic_df: pd.DataFrame = pickle.load(f)

    logger.info("Fetching BACoT answer dataframe...")
    with open("data/bacot_answer.pkl", "rb") as f:
        answer_df: pd.DataFrame = pickle.load(f)

    logger.info("Fetching BACoT NLI info dataframe...")
    with open("data/bacot_nli_info.pkl", "rb") as f:
        nli_df: pd.DataFrame = pickle.load(f)

    logger.info("Fetching BACoT reasoning info dataframe...")
    with open("data/bacot_reasoning_info.pkl", "rb") as f:
        reasoning_df: pd.DataFrame = pickle.load(f)

    logger.info("Merging dataframes...")
    return basic_df.merge(answer_df, on="id").merge(nli_df, on="id").merge(reasoning_df, on="id")


def main():
    logger.info("Starting main metrics calculation...")
    bacot_df: pd.DataFrame = build_bacot_data_from_pickle()

    main_metrics_df: pd.DataFrame = bacot_df[["id"]].copy()

    logger.info("Evaluating Clarity...")
    main_metrics_df["flesch_reading_ease"] = bacot_df["answer"].apply(
        MainMetrics.evaluate_flesch_reading_ease
    )
    main_metrics_df["flesch_kincaid_grade"] = bacot_df["answer"].apply(
        MainMetrics.evaluate_flesch_kincaid_grade
    )

    logger.info("Evaluating Relevance...")
    bert_eval_dict = MainMetrics.evaluate_bert_score(
        candidate=bacot_df["answer"], reference=bacot_df["human_answer"]
    )

    main_metrics_df["precision_bert"] = bert_eval_dict["bert_precision"]
    main_metrics_df["recall_bert"] = bert_eval_dict["bert_recall"]
    main_metrics_df["f1_bert"] = bert_eval_dict["bert_f1"]

    logging.info("Evaluating Factuality...")
    main_metrics_df["factuality_precision_50"] = bacot_df.apply(
        lambda x: MainMetrics.evaluate_factuality(
            facts_to_check=x["fact_ai_list"], fact_reference=x["fact_human_str"]
        ),
        axis=1,
    )

    logging.info("Evaluating Actionability...")
    main_metrics_df["recommendations_precision_50"] = bacot_df.apply(
        lambda x: MainMetrics.evaluate_actionability(
            recommendations_to_check=x["rec_ai_list"], recommendations_reference=x["rec_human_str"]
        ),
        axis=1,
    )

    logging.info("Evaluating Reasoning...")
    main_metrics_df["reasoning_percentage"] = bacot_df["reasoning_dict"].apply(
        lambda x: MainMetrics.evaluate_reasoning
    )

    logger.info("Main metrics calculation completed.")
    logging.info(main_metrics_df)

    # NOTE: you can save main_metrics however you like after this.


if __name__ == "__main__":
    main()
