from typing import List, Union
from models import ModelEncapsulator
from tokenizers import Tokenizer as WPTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    def __init__(self, model_list, tokenizer=None, vectorizer=None):
        """"""
        self.model_map: dict[ModelEncapsulator] = model_list
        self.tokenizer: WPTokenizer = tokenizer
        self.vectorizer: Union[CountVectorizer, TfidfVectorizer] = vectorizer
        self.results = {}

    def evaluate(self, X, y):
        """
        Evaluate all models in the model list.
        Returns a dictionary with model names as keys and their evaluation results as values.
        """
        # record the time taken for each model
        import time

        start_time = time.time()
        model_time_map = {}
        print("🔍 Evaluating models...")
        # for model in self.model_list:
        for key, model in self.model_map.items():
            print(f"Evaluating model: {key}")

            accuracy, report = model.perform_pipeline(X, y)
            self.results[model.name] = report
            model_time_map[model.name] = time.time() - start_time
        return self.results, model_time_map

    def compare_classification_reports(
        self,
        reports: dict,
        timings: dict,
        labels=None,
        metrics=None,
        output_dir: str = "./",
    ):
        """
        Compare multiple classification reports with execution timings and export to CSV.

        Args:
            reports (dict): Dictionary where keys are model names and values are classification_report dicts.
            timings (dict): Dictionary where keys are model names and values are execution time (in seconds).
            labels (list): List of class labels to include. If None, inferred from first report.
            metrics (list): List of metrics to compare. Default: ['precision', 'recall', 'f1-score']
            output_dir (str): Directory to save CSV files (default: current folder).
        """
        if not metrics:
            metrics = ["precision", "recall", "f1-score"]

        first_report = next(iter(reports.values()))
        second_report = next(iter(reports.values()))
        third_report = next(iter(reports.values()))
        if labels is None:
            labels = [
                label
                for label in first_report
                if label not in ["accuracy", "macro avg", "weighted avg"]
            ]

        # 📊 Print per-model results
        for model_name, report in reports.items():
            df = pd.DataFrame(report)
            df = df.transpose().round(3)
            print(f"\n🔍 Results for: {model_name}")
            print(df)
            print("-" * 60)

        # 📉 Class-level comparison
        class_data = []
        for label in labels:
            row = {"class": label}
            for model_name, report in reports.items():
                for metric in metrics:
                    row[f"{model_name}_{metric}"] = report[label][metric]
            class_data.append(row)

        class_df = pd.DataFrame(class_data)

        # 📈 Delta Heatmap between first two models (if applicable)
        if len(reports) >= 2:
            model_names = list(reports.keys())
            base, compare = model_names[0], model_names[1]
            diff_df = class_df[["class"]].copy()
            for metric in metrics:
                diff_df[f"diff_{metric}"] = (
                    class_df[f"{compare}_{metric}"] - class_df[f"{base}_{metric}"]
                )

            print(f"\n📊 Class-Level Deltas ({compare} - {base}):")
            print(diff_df)

            plt.figure(figsize=(8, 4))
            sns.heatmap(
                diff_df.set_index("class")[[f"diff_{m}" for m in metrics]],
                annot=True,
                cmap="coolwarm",
                center=0,
            )
            plt.title(f"🔥 Performance Delta Heatmap ({compare} - {base})")
            plt.tight_layout()
            plt.show()

            # Export deltas to CSV
            diff_df.to_csv(
                f"{output_dir}/model_deltas_{base}_vs_{compare}.csv", index=False
            )

        # ✅ Summary metrics (Accuracy, Macro F1, Weighted F1 + Time)
        summary = {}
        for model_name, report in reports.items():
            summary[model_name] = {
                "Accuracy": report["accuracy"],
                "Macro F1": report["macro avg"]["f1-score"],
                "Weighted F1": report["weighted avg"]["f1-score"],
                "Time (s)": timings.get(model_name, float("nan")),
            }

        summary_df = pd.DataFrame(summary).T.round(3)
        print("\n✅ Overall Summary (with Execution Time):")
        print(summary_df)

        # Export summary and class-level comparison to CSV
        summary_df.to_csv(f"{output_dir}/model_summary.csv")
        class_df.to_csv(f"{output_dir}/class_level_comparison.csv", index=False)

        print("\n🧠 Final Conclusion:")
        best_model = summary_df["Accuracy"].idxmax()
        print(f"→ Best overall accuracy achieved by: **{best_model}**")

        fastest_model = summary_df["Time (s)"].idxmin()
        print(f"→ Fastest model (shortest execution time): **{fastest_model}**")

        print("\n📁 CSVs saved to:", output_dir)
        print("Analysis complete.")
        print("\n" + "-" * 60)
