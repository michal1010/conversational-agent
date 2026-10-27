from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple


EmotionPoint = Tuple[float, float, float]


DEFAULT_EMOTIONS = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
)


def softmax(values: Sequence[float]) -> list[float]:
    """Numerically stable softmax."""
    if not values:
        raise ValueError("softmax() received an empty sequence.")

    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)

    if total == 0.0:
        raise ValueError("Softmax underflow: sum of exponentials is zero.")

    return [v / total for v in exps]


def weighted_euclidean_distance(
    a: EmotionPoint,
    b: EmotionPoint,
    weights: EmotionPoint = (1.0, 1.0, 1.0),
) -> float:
    """
    Weighted Euclidean distance in VAD space.
    weights = (w_valence, w_arousal, w_dominance)
    """
    return math.sqrt(
        weights[0] * (a[0] - b[0]) ** 2
        + weights[1] * (a[1] - b[1]) ** 2
        + weights[2] * (a[2] - b[2]) ** 2
    )


def load_vad_prototypes(
    file_path: str | Path,
    *,
    delimiter: str = ",",
    required_emotions: Iterable[str] | None = DEFAULT_EMOTIONS,
    lowercase_labels: bool = True,
) -> Dict[str, EmotionPoint]:
    """
    Load emotion prototype VAD points from a CSV/text file.

    Expected file format:
        emotion,valence,arousal,dominance
        anger,-0.51,0.59,0.25
        disgust,-0.60,0.35,0.11
        fear,-0.64,0.60,-0.43
        joy,0.76,0.48,0.35
        sadness,-0.63,-0.27,-0.33
        surprise,0.4,0.67,-0.13
        neutral,0.00,0.00,0.00

    Current values are based on English Ekman terms by Mehrabian and Russell' An Approach to Environmental Psychology (1974).
    Big assumption that neutral is indeed at the center of the space.
    Big assumption that computing distances from these scores is a valid way to map from VAD to Ekman categories (no scientific backing).    

    Returns:
        dict[str, tuple[float, float, float]]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Prototype file not found: {file_path}")

    prototypes: Dict[str, EmotionPoint] = {}

    with file_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        required_columns = {"emotion", "valence", "arousal", "dominance"}
        if reader.fieldnames is None:
            raise ValueError("Input file is empty or missing a header row.")

        fieldnames = {name.strip().lower() for name in reader.fieldnames}
        missing_cols = required_columns - fieldnames
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {sorted(missing_cols)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):  # header is line 1
            try:
                emotion = row["emotion"].strip()
                if lowercase_labels:
                    emotion = emotion.lower()

                if not emotion:
                    raise ValueError("emotion label is empty")

                valence = float(row["valence"]) + 1.0 / 2.0  # Convert from [-1, 1] to [0, 1]
                arousal = float(row["arousal"]) + 1.0 / 2.0
                dominance = float(row["dominance"]) + 1.0 / 2.0

                prototypes[emotion] = (valence, arousal, dominance)

            except KeyError as e:
                raise ValueError(f"Malformed row at line {row_idx}: missing {e}") from e
            except ValueError as e:
                raise ValueError(f"Malformed row at line {row_idx}: {e}") from e

    if not prototypes:
        raise ValueError("No prototypes were loaded from the file.")

    if required_emotions is not None:
        required_set = {
            e.lower() if lowercase_labels else e
            for e in required_emotions
        }
        missing = required_set - set(prototypes.keys())
        if missing:
            raise ValueError(
                f"Missing required emotions in prototype file: {sorted(missing)}"
            )

    return prototypes


@dataclass
class VADEmotionMapper:
    prototypes: Mapping[str, EmotionPoint]
    weights: EmotionPoint = (1.0, 1.0, 1.0)
    temperature: float = 0.25

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")

        if len(self.weights) != 3:
            raise ValueError("weights must have exactly 3 values")

        if not self.prototypes:
            raise ValueError("prototypes cannot be empty")

    def distances(self, vad_point: EmotionPoint) -> Dict[str, float]:
        """Return raw distances to each prototype."""
        return {
            emotion: weighted_euclidean_distance(
                vad_point,
                proto,
                weights=self.weights,
            )
            for emotion, proto in self.prototypes.items()
        }

    def predict_proba(self, vad_point: EmotionPoint) -> Dict[str, float]:
        """
        Convert distances to probabilities using softmax over negative distances.

        Smaller distance -> larger logit -> larger probability.
        """
        dists = self.distances(vad_point)
        emotions = list(dists.keys())
        logits = [-dists[e] / self.temperature for e in emotions]
        probs = softmax(logits)
        return dict(zip(emotions, probs))

    def predict(self, vad_point: EmotionPoint) -> str:
        """Return the most likely emotion."""
        probs = self.predict_proba(vad_point)
        return max(probs, key=probs.get)