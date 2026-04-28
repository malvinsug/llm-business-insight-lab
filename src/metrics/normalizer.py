import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


class MetricNormalizer:
    """
    A comprehensive normalizer for text evaluation metrics.

    Handles:
    1. Skewness analysis to determine log transformation needs
    2. Log1p transformation for heavily skewed metrics
    3. Sign alignment (flipping "lower is better" metrics)
    4. Z-score standardization
    5. Composite quality index computation

    Parameters
    ----------
    skewed_metrics : list of str
        Metric names that require log1p transformation.
    lower_is_better : list of str
        Metric names where lower values indicate better quality.
    skew_threshold : float, default 1.0
        Absolute skewness threshold to consider a metric "heavily skewed".
    improvement_threshold : float, default 0.1
        Minimum skewness reduction from log transform to recommend transformation.
    """

    def __init__(
        self,
        skewed_metrics=None,
        lower_is_better=None,
        skew_threshold=1.0,
        improvement_threshold=0.1,
    ):
        self.skewed_metrics = skewed_metrics or []
        self.lower_is_better = lower_is_better or []
        self.skew_threshold = skew_threshold
        self.improvement_threshold = improvement_threshold
        self.scaler_ = None
        self.cols_to_scale_ = None
        self.analysis_results_ = None

        # Full metric list from the original script
        self.all_metrics = [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "clarity_score_avg",
            "f1_bert",
            "relevance_score_avg",
            "precision_bert",
            "recall_bert",
            "factuality_precision_50",
            "recommendations_precision_50",
            "reasoning_percentage",
            "factuality_score_avg",
            "actionability_score_avg",
            "reasoning_score_avg",
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

    def analyze_skewness(self, df, metrics=None):
        """
        Analyze skewness for each metric and recommend log transformations.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing metrics.
        metrics : list of str, optional
            Metrics to analyze. If None, uses all_metrics.

        Returns
        -------
        pd.DataFrame
            Analysis results with skewness metrics and recommendations.
        """
        metrics = metrics or self.all_metrics
        results = []

        for m in metrics:
            if m not in df.columns:
                continue

            data = df[m].dropna()
            if len(data) == 0:
                continue

            orig_skew = skew(data)
            log_data = np.log1p(data)
            log_skew = skew(log_data)

            improvement = abs(orig_skew) - abs(log_skew)
            recommend = (
                "YES"
                if (
                    abs(orig_skew) > self.skew_threshold
                    and improvement > self.improvement_threshold
                )
                else "NO"
            )

            results.append(
                {
                    "Metric": m,
                    "Original Skew": round(orig_skew, 3),
                    "Log1p Skew": round(log_skew, 3),
                    "Skew Reduction": round(improvement, 3),
                    "Min Val": round(data.min(), 5),
                    "Max Val": round(data.max(), 5),
                    "Log Transform?": recommend,
                }
            )

        self.analysis_results_ = pd.DataFrame(results)
        return self.analysis_results_

    def fit(self, df):
        """
        Fit the normalizer on the data.

        Analyzes skewness, applies transformations, and fits the StandardScaler.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing metrics.

        Returns
        -------
        self
        """
        # Stage 1: Skewness analysis (auto-detect if not provided)
        if not self.skewed_metrics:
            analysis = self.analyze_skewness(df)
            self.skewed_metrics = analysis[analysis["Log Transform?"] == "YES"]["Metric"].tolist()

        # Make a copy to avoid modifying original
        df_work = df.copy()

        # Stage 2: Log transform for skewed metrics
        for m in self.skewed_metrics:
            if m in df_work.columns:
                df_work[f"{m}_log_transformed"] = np.log1p(df_work[m])
            else:
                self.skewed_metrics.remove(m)

        # Stage 3: Alignment (sign flipping)
        for m in self.lower_is_better:
            if m in df_work.columns:
                df_work[f"{m}_aligned"] = df_work[m] * -1

        # Stage 4: Build column list for scaling
        standard_metrics = [
            m
            for m in self.all_metrics
            if m not in self.skewed_metrics and m not in self.lower_is_better
        ]

        self.cols_to_scale_ = (
            [f"{m}_log_transformed" for m in self.skewed_metrics]
            + [f"{m}_aligned" for m in self.lower_is_better]
            + [m for m in standard_metrics if m in df_work.columns]
        )

        # Stage 5: Fit scaler
        self.scaler_ = StandardScaler()
        self.scaler_.fit(df_work[self.cols_to_scale_])

        return self

    def transform(self, df):
        """
        Transform the data using the fitted normalizer.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing metrics.

        Returns
        -------
        pd.DataFrame
            Normalized dataframe with z-scored metrics and quality index.
        """
        if self.scaler_ is None:
            raise ValueError("Normalizer must be fitted before transform. Call fit() first.")

        df_work = df.copy()

        # Apply log transforms
        for m in self.skewed_metrics:
            if m in df_work.columns:
                df_work[f"{m}_log_transformed"] = np.log1p(df_work[m])

        # Apply alignment
        for m in self.lower_is_better:
            if m in df_work.columns:
                df_work[f"{m}_aligned"] = df_work[m] * -1

        # Scale
        scaled_data = self.scaler_.transform(df_work[self.cols_to_scale_])

        # Build output dataframe
        norm_cols = [
            f'norm_{c.replace("_log_transformed", "").replace("_aligned", "")}'
            for c in self.cols_to_scale_
        ]
        df_normalized = pd.DataFrame(scaled_data, columns=norm_cols, index=df.index)

        # Composite quality index
        df_normalized["overall_quality_index"] = df_normalized.mean(axis=1)
        df_normalized["id"] = df["id"].copy()

        return df_normalized

    def fit_transform(self, df):
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing metrics.

        Returns
        -------
        pd.DataFrame
            Normalized dataframe.
        """
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self):
        """Return the names of normalized features."""
        if self.cols_to_scale_ is None:
            raise ValueError("Normalizer must be fitted first.")
        return [
            f'norm_{c.replace("_log_transformed", "").replace("_aligned", "")}'
            for c in self.cols_to_scale_
        ]

    def get_analysis_summary(self):
        """Return the skewness analysis results."""
        if self.analysis_results_ is None:
            raise ValueError("No analysis performed yet. Call analyze_skewness() or fit() first.")
        return self.analysis_results_
