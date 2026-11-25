"""Runs inference over a list of cases off the main thread, with cancel and resume."""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

from unet_inference_client.backend import Backend


@dataclass(frozen=True)
class BatchPlan:
    to_run: list[str]
    skipped: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BatchSummary:
    completed: int
    skipped: int
    failed: int
    cancelled: bool
    remaining: list[str]


def plan(case_ids: list[str], done: set[str], skip_existing: bool) -> BatchPlan:
    if not skip_existing:
        return BatchPlan(list(case_ids))
    return BatchPlan(
        [case for case in case_ids if case not in done],
        [case for case in case_ids if case in done],
    )


class BatchWorker(QObject):
    progress = Signal(int, int, str)
    case_done = Signal(str, float)
    case_failed = Signal(str, str)
    finished = Signal(object)

    def __init__(self, backend: Backend, batch: BatchPlan):
        super().__init__()
        self.backend = backend
        self.batch = batch
        self.cancelled = False
        self.attempted = 0

    def cancel(self) -> None:
        self.cancelled = True

    @property
    def remaining(self) -> list[str]:
        return self.batch.to_run[self.attempted:]

    def run(self) -> None:
        completed = failed = 0
        total = len(self.batch.to_run)
        for index, case_id in enumerate(self.batch.to_run, start=1):
            if self.cancelled:
                break
            self.attempted += 1
            self.progress.emit(index, total, case_id)
            try:
                result = self.backend.run_inference(case_id)
                completed += 1
                self.case_done.emit(case_id, result.elapsed_seconds)
            except Exception as exc:
                failed += 1
                self.case_failed.emit(case_id, str(exc))
        self.finished.emit(BatchSummary(
            completed, len(self.batch.skipped), failed, self.cancelled, self.remaining
        ))
