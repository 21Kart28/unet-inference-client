"""Dice per case for two models run on the same cases, and the difference between them."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from unet_inference_client import metrics
from unet_inference_client.history import StoredResult

CSV_FIELDS = ["case_id", "model_a", "dice_a", "model_b", "dice_b", "delta"]


@dataclass(frozen=True)
class DeltaRow:
    case_id: str
    dice_a: float | None
    dice_b: float | None

    @property
    def delta(self) -> float | None:
        if self.dice_a is None or self.dice_b is None:
            return None
        return self.dice_b - self.dice_a


@dataclass(frozen=True)
class DeltaSummary:
    paired: int
    better: int
    worse: int
    mean_delta: float | None


def dice_by_model(results: list[StoredResult]) -> dict[str, dict[str, float]]:
    """model -> case -> Dice, from saved metrics; the newest result for a case wins."""
    table: dict[str, dict[str, tuple[str, float]]] = {}
    for result in results:
        saved = metrics.load_metrics(result.output_dir)
        if saved is None or saved.get("dice") is None:
            continue
        cases = table.setdefault(result.model, {})
        previous = cases.get(result.case_id)
        if previous is None or result.timestamp >= previous[0]:
            cases[result.case_id] = (result.timestamp, float(saved["dice"]))
    return {model: {case: dice for case, (_, dice) in cases.items()} for model, cases in table.items()}


def delta_rows(results: list[StoredResult], model_a: str, model_b: str) -> list[DeltaRow]:
    table = dice_by_model(results)
    dice_a, dice_b = table.get(model_a, {}), table.get(model_b, {})
    return [
        DeltaRow(case, dice_a.get(case), dice_b.get(case))
        for case in sorted(set(dice_a) | set(dice_b))
    ]


def summarize(rows: list[DeltaRow]) -> DeltaSummary:
    deltas = [row.delta for row in rows if row.delta is not None]
    if not deltas:
        return DeltaSummary(0, 0, 0, None)
    return DeltaSummary(
        paired=len(deltas),
        better=sum(delta > 0 for delta in deltas),
        worse=sum(delta < 0 for delta in deltas),
        mean_delta=sum(deltas) / len(deltas),
    )


def describe(summary: DeltaSummary, model_a: str, model_b: str) -> str:
    if summary.paired == 0:
        return (f"No case has saved Dice for both {model_a} and {model_b}."
                " Run Compare all results for each model first.")
    direction = "better" if summary.mean_delta > 0 else "worse" if summary.mean_delta < 0 else "equal"
    return (f"{summary.paired} paired case(s): {model_b} is {direction} on average"
            f" ({summary.mean_delta:+.4f} Dice), better on {summary.better}, worse on {summary.worse}.")


def write_delta_csv(path: Path, rows: list[DeltaRow], model_a: str, model_b: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "case_id": row.case_id,
                "model_a": model_a, "dice_a": row.dice_a,
                "model_b": model_b, "dice_b": row.dice_b,
                "delta": row.delta,
            })
