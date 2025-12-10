import os, sys, argparse
import pandas as pd
from core import HateModelInference
from utils import resolve_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferring Hate Speech on a CSV")
    parser.add_argument("--input", required=True, help="Path of the CSV file or just filename")
    parser.add_argument("--output", required=False, help="CSV output path with label")

    args = parser.parse_args()

    input_path = resolve_path(args.input)

    if args.output:
        output_path = args.output
    else:
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        output_dir = os.path.join(root, "results")
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.splitext(os.path.basename(input_path))[0] + "_pred.csv"
        output_path = os.path.join(output_dir, fname)
    
    df = pd.read_csv(input_path)
  
    if df.shape[1] != 1:
      raise ValueError("The CSV must have a single column containing comments.")

    comments = df.iloc[:, 0].astype(str).tolist()
    
    model = HateModelInference()

    predictions = []

    for text in comments:
        predictions.append(model.predict_text(text))

    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    df_out = pd.DataFrame(predictions, columns=labels)
    df_out.insert(0, "comment", comments)  

    df_out.to_csv(output_path, index=False)
    print(f"File saved in: {output_path}")

