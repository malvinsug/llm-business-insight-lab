import pickle
import pandas as pd


from src.metrics.normalizer import MetricNormalizer

SKEWED_METRICS = [
    "answer_hedge_to_booster_ratio",
    "recommendations_precision_50",
    "answer_quantification_density",
    "human_answer_quantification_density",
    "factuality_precision_50",
]

LOWER_IS_BETTER = ["flesch_kincaid_grade"]


def load_bacot_metrics():
    with open("./data/bacot_all_metrics.pkl", "rb") as f:
        bacot_metrics: pd.DataFrame = pickle.load(f)

    return bacot_metrics


def main():

    df: pd.DataFrame = load_bacot_metrics()

    normalizer: MetricNormalizer = MetricNormalizer(
        skewed_metrics=SKEWED_METRICS,  # Metrics needing log transform
        lower_is_better=LOWER_IS_BETTER,  # Metrics to flip
        skew_threshold=1.0,  # Skewness threshold for auto-detection
        improvement_threshold=0.1,  # Min improvement to recommend transform
    )

    normalizer.fit(df)

    norm_df: pd.DataFrame = normalizer.transform(df)

    # NOTE: you can save the normalized metrics however you like after this.


if __name__ == "__main__":
    main()
