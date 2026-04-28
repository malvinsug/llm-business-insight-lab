import pandas as pd
from src.metrics.reference_free_metrics import ReferenceFreeMetrics
import logging
import pickle

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import transformers

transformers.utils.logging.set_verbosity_error()


def build_bacot_data_from_pickle():
    logger.info("Fetching BACoT answer dataframe...")
    with open("data/bacot_answer.pkl", "rb") as f:
        answer_df: pd.DataFrame = pickle.load(f)

    return answer_df


def main():
    logger.info("Starting reference free metrics calculation...")
    bacot_df: pd.DataFrame = build_bacot_data_from_pickle()

    reference_free_metrics_df: pd.DataFrame = bacot_df[["id"]].copy()

    logger.info("Evaluating Quantification Density...")
    reference_free_metrics_df["answer_quantification_density"] = bacot_df["answer"].apply(
        lambda txt: ReferenceFreeMetrics.quantification_density(txt)["quantification_density"]
    )

    reference_free_metrics_df["human_answer_quantification_density"] = bacot_df[
        "human_answer"
    ].apply(lambda txt: ReferenceFreeMetrics.quantification_density(txt)["quantification_density"])

    logger.info("Evaluating Causal Connective Density...")
    reference_free_metrics_df["answer_causal_density"] = bacot_df["answer"].apply(
        lambda txt: ReferenceFreeMetrics.causal_connective_density(txt)["causal_connective_density"]
    )
    reference_free_metrics_df["human_answer_causal_density"] = bacot_df["human_answer"].apply(
        lambda txt: ReferenceFreeMetrics.causal_connective_density(txt)["causal_connective_density"]
    )

    logger.info("Evaluating Hedging Calibration...")
    reference_free_metrics_df["answer_hedging_calibration"] = bacot_df["answer"].apply(
        ReferenceFreeMetrics.hedging_calibration
    )
    reference_free_metrics_df["human_answer_hedging_calibration"] = bacot_df["human_answer"].apply(
        ReferenceFreeMetrics.hedging_calibration
    )
    reference_free_metrics_df["answer_hedge_to_booster_ratio"] = reference_free_metrics_df[
        "answer_hedging_calibration"
    ].apply(lambda obj: obj["hedge_to_booster_ratio"])
    reference_free_metrics_df["human_answer_hedge_to_booster_ratio"] = reference_free_metrics_df[
        "human_answer_hedging_calibration"
    ].apply(lambda obj: obj["hedge_to_booster_ratio"])
    reference_free_metrics_df["answer_hedge_density"] = reference_free_metrics_df[
        "answer_hedging_calibration"
    ].apply(lambda obj: obj["hedge_density"])
    reference_free_metrics_df["human_answer_hedge_density"] = reference_free_metrics_df[
        "human_answer_hedging_calibration"
    ].apply(lambda obj: obj["hedge_density"])
    reference_free_metrics_df["answer_booster_density"] = reference_free_metrics_df[
        "answer_hedging_calibration"
    ].apply(lambda obj: obj["booster_density"])
    reference_free_metrics_df["human_answer_booster_density"] = reference_free_metrics_df[
        "human_answer_hedging_calibration"
    ].apply(lambda obj: obj["booster_density"])

    logger.info("Evaluating Modal Verb Profile...")
    reference_free_metrics_df["answer_modal_verb_profile"] = bacot_df["answer"].apply(
        ReferenceFreeMetrics.modal_verb_profile
    )
    reference_free_metrics_df["human_answer_modal_verb_profile"] = bacot_df["answer"].apply(
        ReferenceFreeMetrics.modal_verb_profile
    )
    reference_free_metrics_df["answer_modal_should_freq"] = reference_free_metrics_df[
        "answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_should_freq"])
    reference_free_metrics_df["human_answer_modal_should_freq"] = reference_free_metrics_df[
        "human_answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_should_freq"])
    reference_free_metrics_df["answer_modal_must_freq"] = reference_free_metrics_df[
        "answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_must_freq"])
    reference_free_metrics_df["human_answer_modal_must_freq"] = reference_free_metrics_df[
        "human_answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_must_freq"])
    reference_free_metrics_df["answer_modal_could_freq"] = reference_free_metrics_df[
        "answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_could_freq"])
    reference_free_metrics_df["human_answer_modal_could_freq"] = reference_free_metrics_df[
        "human_answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_could_freq"])
    reference_free_metrics_df["answer_modal_will_freq"] = reference_free_metrics_df[
        "answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_will_freq"])
    reference_free_metrics_df["human_answer_modal_will_freq"] = reference_free_metrics_df[
        "human_answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_will_freq"])
    reference_free_metrics_df["answer_modal_may_freq"] = reference_free_metrics_df[
        "answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_may_freq"])
    reference_free_metrics_df["human_answer_modal_may_freq"] = reference_free_metrics_df[
        "human_answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_may_freq"])
    reference_free_metrics_df["answer_modal_might_freq"] = reference_free_metrics_df[
        "answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_might_freq"])
    reference_free_metrics_df["human_answer_modal_might_freq"] = reference_free_metrics_df[
        "human_answer_modal_verb_profile"
    ].apply(lambda obj: obj["modal_might_freq"])

    reference_free_metrics_df = reference_free_metrics_df[
        [
            "id",
            "answer_quantification_density",
            "human_answer_quantification_density",
            "answer_causal_density",
            "human_answer_causal_density",
            "answer_hedge_to_booster_ratio",
            "human_answer_hedge_to_booster_ratio",
            "answer_hedge_density",
            "human_answer_hedge_density",
            "answer_booster_density",
            "human_answer_booster_density",
            "answer_modal_should_freq",
            "human_answer_modal_should_freq",
            "answer_modal_must_freq",
            "human_answer_modal_must_freq",
            "answer_modal_could_freq",
            "human_answer_modal_could_freq",
            "answer_modal_will_freq",
            "human_answer_modal_will_freq",
            "answer_modal_may_freq",
            "human_answer_modal_may_freq",
            "answer_modal_might_freq",
            "human_answer_modal_might_freq",
        ]
    ]

    logger.info("Reference free metrics calculation completed.")
    logging.info(reference_free_metrics_df)

    # NOTE: you can save main_metrics however you like after this.


if __name__ == "__main__":
    main()
