from pathlib import Path

from unet_inference_client import batch
from unet_inference_client.backend import Backend, InferenceResult


class RecordingBackend(Backend):
    def __init__(self, fail=(), on_run=None):
        self.ran = []
        self.fail = set(fail)
        self.on_run = on_run

    def run_inference(self, case_id):
        self.ran.append(case_id)
        if self.on_run:
            self.on_run(case_id)
        if case_id in self.fail:
            raise RuntimeError(f"{case_id} broke")
        return InferenceResult(case_id, Path("."), Path("pred.nrrd"), 1.5, "1", "0.9")


def run_worker(worker):
    events = []
    worker.progress.connect(lambda i, n, c: events.append(("start", c)))
    worker.case_done.connect(lambda c, s: events.append(("done", c)))
    worker.case_failed.connect(lambda c, m: events.append(("failed", c, m)))
    summaries = []
    worker.finished.connect(summaries.append)
    worker.run()
    return events, summaries[0]


def test_plan_skips_cases_already_done():
    result = batch.plan(["A", "B", "C"], {"B"}, skip_existing=True)
    assert result.to_run == ["A", "C"]
    assert result.skipped == ["B"]


def test_plan_runs_everything_when_not_skipping():
    result = batch.plan(["A", "B"], {"B"}, skip_existing=False)
    assert result.to_run == ["A", "B"]
    assert result.skipped == []


def test_worker_runs_each_case_and_reports_failures(qt_app):
    backend = RecordingBackend(fail={"B"})
    events, summary = run_worker(batch.BatchWorker(backend, batch.BatchPlan(["A", "B"], ["Z"])))

    assert backend.ran == ["A", "B"]
    assert events == [("start", "A"), ("done", "A"), ("start", "B"), ("failed", "B", "B broke")]
    assert summary == batch.BatchSummary(1, 1, 1, False, [])


def test_cancel_stops_between_cases_and_keeps_the_rest(qt_app):
    worker = None
    backend = RecordingBackend(on_run=lambda case: worker.cancel() if case == "A" else None)
    worker = batch.BatchWorker(backend, batch.BatchPlan(["A", "B", "C"]))
    events, summary = run_worker(worker)

    assert backend.ran == ["A"]
    assert events == [("start", "A"), ("done", "A")]
    assert summary.cancelled
    assert summary.remaining == ["B", "C"]
    assert summary.completed == 1
