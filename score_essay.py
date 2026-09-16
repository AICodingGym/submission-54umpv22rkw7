"""Score English essays with the locally trained, calibrated LightGBM model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from baseline import integer_scores


DEFAULT_MODEL = Path(__file__).resolve().parent / "outputs_tuned/model.joblib"


class EssayScorer:
    """Load a trusted local model once and reuse it for multiple requests."""

    def __init__(self, model_path=DEFAULT_MODEL):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.center = bundle["center"]
        self.scale = bundle["scale"]
        self.offset = bundle["offset"]

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            raise ValueError("Provide at least one essay")
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Essay at position {index + 1} must be non-empty text")
        raw = np.asarray(self.pipeline.predict(texts), dtype=float)
        calibrated = self.center + self.scale * (raw - self.center) + self.offset
        if not np.isfinite(calibrated).all():
            raise ValueError("Model produced a non-finite score")
        # Continuous values are model outputs, not confidence estimates.
        return integer_scores(calibrated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="One English essay")
    source.add_argument("--file", type=Path, help="UTF-8 text file containing one essay")
    source.add_argument("--csv", type=Path, help="CSV with essay_id and full_text columns")
    parser.add_argument("--output", type=Path, help="Output file; defaults to stdout")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        source_path = args.csv or args.file
        if args.output and (
            args.output.resolve() == args.model.resolve()
            or (source_path and args.output.resolve() == source_path.resolve())
        ):
            raise ValueError("Output must not overwrite the input or model")
        if args.csv:
            frame = pd.read_csv(args.csv, dtype={"essay_id": str}, keep_default_na=False)
            if not {"essay_id", "full_text"}.issubset(frame.columns):
                raise ValueError("CSV must contain essay_id and full_text columns")
            if not frame.essay_id.is_unique or frame.essay_id.str.strip().eq("").any():
                raise ValueError("Essay IDs must be non-empty and unique")
            texts = frame.full_text.tolist()
        else:
            texts = [args.file.read_text(encoding="utf-8") if args.file else args.text]
        scores = EssayScorer(args.model).predict(texts)
        if args.csv:
            output = frame[["essay_id"]].assign(score=scores).to_csv(index=False)
        else:
            output = json.dumps({"score": int(scores[0]), "score_range": [1, 6]}) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
