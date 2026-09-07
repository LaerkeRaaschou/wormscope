"""SAM2 on a background thread, so the GUI stays usable while it thinks.

Every call into the segmenter — building it, adding a prompt, tracking,
pre-encoding — is queued here and executed by one worker thread. That has two
consequences worth stating plainly:

* **The main thread never touches the backend.** SAM2 is not thread-safe, so
  serialising every call through a single thread is what makes this safe. The
  GUI only ever sees results arriving as Qt signals.
* **You can keep working while it runs.** Painting, stepping frames, undo —
  none of it waits for inference any more.

Jobs run in the order they were queued. A tracking job checks a stop flag
between frames, so Stop takes effect within one frame rather than at the end.
"""

import atexit
import queue
import threading
import traceback
import weakref

from PyQt6.QtCore import QThread, pyqtSignal

from ..segmentation import MissingSam2Paths

# Every live worker, so the interpreter can never be left waiting on one. A
# QThread blocked on its queue would otherwise hang process exit.
_LIVE: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _stop_all_workers():
    for worker in list(_LIVE):
        try:
            worker.shutdown()
        except Exception:                              # pragma: no cover
            pass


class SamWorker(QThread):
    """One thread, one job queue, one segmenter."""

    # backend lifecycle
    built = pyqtSignal(bool, str)                  # ok, error message
    needsFiles = pyqtSignal(str)                   # no config/checkpoint yet
    # prompt results
    promptDone = pyqtSignal(int, int, object, str)  # obj_id, frame, mask, kind
    # tracking
    frameTracked = pyqtSignal(int, int, object)    # obj_id, frame, mask
    # generic
    jobProgress = pyqtSignal(int, int, str)        # done, total, detail
    jobDone = pyqtSignal(str, object)              # kind, payload
    jobFailed = pyqtSignal(str, str)               # kind, error

    def __init__(self, factory, frames, shape, parent=None):
        super().__init__(parent)
        self._factory = factory                    # () -> segmenter backend
        self._frames = frames
        self._shape = shape
        self._jobs: queue.Queue = queue.Queue()
        self._stop_job = threading.Event()         # "abandon the running job"
        self._quit = threading.Event()
        self.backend = None                        # only touched in this thread
        self._busy = threading.Event()
        _LIVE.add(self)

    # --- queueing (called from the GUI thread) -------------------------------
    def submit(self, kind, fn):
        """Queue `fn(backend)`; its return value is emitted with jobDone."""
        self._jobs.put((kind, fn))

    def build(self):
        """Create the backend (loads the model — slow) without doing anything
        else. Emits built(ok, message)."""
        self._jobs.put(("build", None))

    def add_points(self, frame_idx, obj_id, points, labels):
        def job(backend):
            mask = backend.add_points(frame_idx, obj_id, points, labels)
            self.promptDone.emit(int(obj_id), int(frame_idx), mask, "points")
            return mask
        self.submit("points", job)

    def add_mask(self, frame_idx, obj_id, mask):
        def job(backend):
            out = backend.add_mask(frame_idx, obj_id, mask)
            self.promptDone.emit(int(obj_id), int(frame_idx), out, "mask")
            return out
        self.submit("mask", job)

    def track(self, plan, total):
        """`plan` is a list of (obj_id, anchors, runs):

        * anchors — [(frame, mask)] fed as mask prompts before tracking, so
          hand corrections survive and become anchors;
        * runs — [(start_frame_idx, reverse)] passed to the backend.
        """
        def job(backend):
            done = 0
            for oid, anchors, runs in plan:
                for frame_idx, mask in anchors:
                    if self._stop_job.is_set():
                        return done
                    backend.add_mask(frame_idx, oid, mask)
                for start, reverse in runs:
                    for frame_idx, mask in backend.propagate(
                            oid, start_frame_idx=start, reverse=reverse):
                        self.frameTracked.emit(int(oid), int(frame_idx), mask)
                        done += 1
                        self.jobProgress.emit(done, total,
                                              f"worm {oid} — frame {frame_idx}")
                        if self._stop_job.is_set():
                            return done
            return done
        self.submit("track", job)

    def prewarm(self, indices):
        def job(backend):
            total = len(indices)

            def progress(done, tot, seconds, bytes_each):
                self.jobProgress.emit(done, tot or total,
                                      f"encoded {done}/{tot or total} "
                                      f"({seconds:.2f}s each)")
                return self._stop_job.is_set()
            return backend.prewarm(indices, progress)
        self.submit("prewarm", job)

    def forget(self, obj_id):
        self.submit("forget", lambda backend: backend.forget(obj_id))

    def reset(self):
        self.submit("reset", lambda backend: backend.reset())

    # --- control -------------------------------------------------------------
    @property
    def busy(self):
        return self._busy.is_set() or not self._jobs.empty()

    def stop_current(self):
        """Ask the running job to give up at the next frame."""
        self._stop_job.set()

    def shutdown(self):
        """Stop the thread and wait for it. Safe to call more than once."""
        if not self.isRunning():
            return
        self._quit.set()
        self._stop_job.set()
        self._jobs.put((None, None))               # unblock a waiting get()
        if not self.wait(5000):                    # pragma: no cover
            print("[sam2] worker did not stop in time")
        _LIVE.discard(self)

    # --- the thread ----------------------------------------------------------
    def run(self):
        while not self._quit.is_set():
            try:
                # Poll rather than block forever, so quitting is prompt even if
                # no sentinel arrives (interpreter shutdown, killed window).
                kind, fn = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind is None or self._quit.is_set():
                break
            self._busy.set()
            self._stop_job.clear()
            try:
                if self.backend is None:
                    try:
                        self.backend = self._factory(self._frames, self._shape)
                    except MissingSam2Paths as err:
                        # Not an error to report as a failure: the GUI answers
                        # this by opening the file picker.
                        self.needsFiles.emit(str(err))
                        continue
                    self.built.emit(True, "")
                if kind == "build":
                    self.jobDone.emit(kind, None)
                else:
                    self.jobDone.emit(kind, fn(self.backend))
            except Exception as err:               # noqa: BLE001 - reported to the GUI
                traceback.print_exc()
                if self.backend is None:
                    self.built.emit(False, str(err))
                self.jobFailed.emit(kind, str(err))
            finally:
                self._busy.clear()
