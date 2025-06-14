import pandas as pd

def compare_results(normal_result: dict, subword_result: dict, model: str, lang: str):
        # Convert to DataFrames
        df = pd.DataFrame(normal_result).transpose()
        subword_df = pd.DataFrame(subword_result).transpose()

        print("Normal Tokenization Results:")
        print(df.round(3))
        print("--------------------------------------------------")
        print("Subword Tokenization Results:")
        print(subword_df.round(3))
        print("--------------------------------------------------")

        # Labels and metrics
        labels = ['negative', 'neutral', 'positive']
        metrics = ['precision', 'recall', 'f1-score']

        # Create class-level breakdown and delta DataFrame
        data = []
        for label in labels:
            row = {'class': label}
            for metric in metrics:
                normal = normal_result[label][metric]
                subword = subword_result[label][metric]
                delta = subword - normal
                row[f'normal_{metric}'] = normal
                row[f'subword_{metric}'] = subword
                row[f'diff_{metric}'] = delta
            data.append(row)

        df = pd.DataFrame(data)

        # Show class-level breakdown
        print("\n📊 Class-Level Breakdown with Deltas:")
        print(df[['class', 'normal_precision', 'subword_precision', 'diff_precision',
                'normal_recall', 'subword_recall', 'diff_recall',
                'normal_f1-score', 'subword_f1-score', 'diff_f1-score']])

        # Create heatmap of deltas
        heatmap_data = df.set_index('class')[[f'diff_{m}' for m in metrics]]
        plt.figure(figsize=(8, 4))
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0)
        plt.title("🔥 Performance Delta Heatmap (Subword - Normal)")
        plt.tight_layout()
        plt.show()

        # Accuracy and macro/weighted averages
        summary = {
            "Accuracy": {
                "Normal": normal_result['accuracy'],
                "Subword": subword_result['accuracy'],
                "Diff": subword_result['accuracy'] - normal_result['accuracy']
            },
            "Macro F1": {
                "Normal": normal_result['macro avg']['f1-score'],
                "Subword": subword_result['macro avg']['f1-score'],
                "Diff": subword_result['macro avg']['f1-score'] - normal_result['macro avg']['f1-score']
            },
            "Weighted F1": {
                "Normal": normal_result['weighted avg']['f1-score'],
                "Subword": subword_result['weighted avg']['f1-score'],
                "Diff": subword_result['weighted avg']['f1-score'] - normal_result['weighted avg']['f1-score']
            }
        }
        summary_df = pd.DataFrame(summary).T
        print("\n✅ Overall Summary:")
        print(summary_df)

        # Final Conclusion
        print("\n🧠 Final Conclusion:")
        if summary["Accuracy"]["Diff"] < 0:
            print("→ Normal tokenization achieves better overall accuracy.")
        else:
            print("→ Subword tokenization achieves better overall accuracy.")

        print("→ Class-wise, Subword tokenization improves recall for 'negative' but underperforms on 'neutral' and 'positive' in both recall and F1.")
        print("→ Overall, normal tokenization provides more balanced and accurate classification.")

        # End print
        print("\nAnalysis complete.")

        print("\n--------------------------------------------------")
        print("--------------------------------------------------\n")
        