"""Markdown and HTML reports of a batch's metrics or a two-model comparison."""

from __future__ import annotations

import html
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from unet_inference_client import metrics
from unet_inference_client.history import StoredResult
from unet_inference_client.model_compare import DeltaRow, DeltaSummary, describe

FILE_FILTER = "Markdown (*.md);;HTML (*.html *.htm)"


@dataclass(frozen=True)
class Table:
    title: str
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class Report:
    title: str
    details: list[tuple[str, str]]
    tables: list[Table]
    notes: list[str] = field(default_factory=list)


def summary_stats(values: list[float]) -> list[tuple[str, str]]:
    if not values:
        return []
    return [
        ("Cases", str(len(values))),
        ("Mean Dice", f"{statistics.fmean(values):.4f}"),
        ("Median Dice", f"{statistics.median(values):.4f}"),
        ("Min Dice", f"{min(values):.4f}"),
        ("Max Dice", f"{max(values):.4f}"),
    ]


def from_results(results: list[StoredResult], server: str, model: str) -> Report:
    """Metrics of every stored result that has a metrics.json."""
    case_rows, class_rows, dice_values = [], [], []
    for result in results:
        saved = metrics.load_metrics(result.output_dir)
        if saved is None:
            continue
        dice = saved.get("dice")
        if dice is not None:
            dice_values.append(float(dice))
        case_rows.append([
            result.case_id, result.model, result.epoch, _number(dice, 4),
            _number(saved.get("pred_volume_mm3"), 1), _number(saved.get("gt_volume_mm3"), 1),
            _number(saved.get("volume_diff_pct"), 2), _number(saved.get("hd95_mm"), 2),
        ])
        for row in saved.get("classes") or []:
            class_rows.append([
                result.case_id, str(row.get("label", "")), _number(row.get("dice"), 4),
                _number(row.get("pred_volume_mm3"), 1), _number(row.get("gt_volume_mm3"), 1),
            ])

    tables = [Table(
        "Per-case metrics",
        ["Case", "Model", "Epoch", "Dice", "Pred vol (mm³)", "GT vol (mm³)", "Vol diff (%)", "HD95 (mm)"],
        case_rows,
    )]
    if class_rows:
        tables.append(Table(
            "Per-class metrics", ["Case", "Label", "Dice", "Pred vol (mm³)", "GT vol (mm³)"], class_rows
        ))
    notes = [] if case_rows else ["No result has saved metrics. Run Compute or Compare all results first."]
    return Report(
        title="Batch metrics report",
        details=[("Server", server), ("Model", model), ("Generated", _now())] + summary_stats(dice_values),
        tables=tables,
        notes=notes,
    )


def from_deltas(rows: list[DeltaRow], summary: DeltaSummary, model_a: str, model_b: str, server: str) -> Report:
    table = Table(
        "Dice per case",
        ["Case", f"{model_a} Dice", f"{model_b} Dice", "Delta"],
        [[row.case_id, _number(row.dice_a, 4), _number(row.dice_b, 4), _signed(row.delta)] for row in rows],
    )
    details = [("Server", server), ("Model A", model_a), ("Model B", model_b), ("Generated", _now())]
    if summary.mean_delta is not None:
        details += [
            ("Paired cases", str(summary.paired)),
            ("Mean delta", _signed(summary.mean_delta)),
            (f"{model_b} better on", str(summary.better)),
            (f"{model_b} worse on", str(summary.worse)),
        ]
    return Report(
        title="Model comparison report",
        details=details,
        tables=[table],
        notes=[describe(summary, model_a, model_b)],
    )


def to_markdown(report: Report) -> str:
    lines = [f"# {report.title}", ""]
    lines += [f"- **{key}:** {value}" for key, value in report.details]
    for table in report.tables:
        lines += ["", f"## {table.title}", ""]
        if not table.rows:
            lines.append("_No rows._")
            continue
        lines.append("| " + " | ".join(table.headers) + " |")
        lines.append("|" + "|".join(" --- " for _ in table.headers) + "|")
        lines += ["| " + " | ".join(cell or "—" for cell in row) + " |" for row in table.rows]
    if report.notes:
        lines += ["", "## Notes", ""]
        lines += [f"- {note}" for note in report.notes]
    return "\n".join(lines) + "\n"


def to_html(report: Report) -> str:
    parts = [
        "<!DOCTYPE html>", "<html lang=\"en\"><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(report.title)}</title>",
        "<style>", _CSS, "</style></head><body>",
        f"<h1>{html.escape(report.title)}</h1>", "<dl>",
    ]
    parts += [f"<dt>{html.escape(key)}</dt><dd>{html.escape(value)}</dd>" for key, value in report.details]
    parts.append("</dl>")
    for table in report.tables:
        parts.append(f"<h2>{html.escape(table.title)}</h2>")
        if not table.rows:
            parts.append("<p class=\"dim\">No rows.</p>")
            continue
        parts.append("<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in table.headers))
        parts.append("</tr></thead><tbody>")
        for row in table.rows:
            parts.append("<tr>" + "".join(f"<td>{html.escape(cell or '—')}</td>" for cell in row) + "</tr>")
        parts.append("</tbody></table>")
    if report.notes:
        parts.append("<h2>Notes</h2><ul>")
        parts += [f"<li>{html.escape(note)}</li>" for note in report.notes]
        parts.append("</ul>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def write(path: Path, report: Report) -> None:
    """Write Markdown or HTML depending on the file suffix; anything else is Markdown."""
    text = to_html(report) if path.suffix.lower() in (".html", ".htm") else to_markdown(report)
    path.write_text(text, encoding="utf-8")


def _number(value: float | str | None, digits: int) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _signed(value: float | None) -> str:
    return "" if value is None else f"{value:+.4f}"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 70rem; color: #222; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; }
dt { font-weight: 600; }
dd { margin: 0; }
table { border-collapse: collapse; margin-bottom: 1.5rem; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead { background: #f3f3f3; }
.dim { color: #777; }
"""
