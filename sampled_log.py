import os
import random

import fastparquet
import pandas as pd

import wandb


TARGET_WANDB_LOG_CALLS = 100000

MIN_STEPS_PER_HISTORY_CHUNK = 100000
MAX_HISTORY_CHUNKS = 100


class SampledWandbLogger(object):
    """Downsamples calls to wandb.log, but saves permanent record of all calls.

    This class can be used to improve wandb frontend performance, by limiting the size
    of data saved to the wandb metrics store via wandb.log.

    In addition to downsampling wandb.log() calls, all wandb.log() calls all stored in a chunked
    parquet table and sync'd as an Artifact. You can use W&B Artifacts APIs to grab the entire
    dataset later if you need access to unsampled metrics.

    This is designed to be future compatible. We are working on a faster and more scalable metrics
    store implementation built on top of chunked parquet files. In the future you will be able to
    use the W&B UI to plot metrics stored in the parquet tables generated by SampledWandbLogger.

    VERY IMPORTANT CAVEAT: This class downsamples by dropping data points at the target rate. You
    should only use it if you have a single set of keys that you pass to every wandb.log() call
    in your program. For example, using SampledWandbLogger will produce unwanted effects if you
    log a set of metrics on each training step, and a different set of metrics at the end of
    each epoch. All wandb.log() calls are sampled at the same rate, even for sets of metrics whose
    relative logging frequency is low. This is probably not the behavior you want in that case.

    CAVEAT #2: You must call run.finish() explicitly when using SampledWandbLogger!

    Usage:
        run = wandb.init(project='sampled-log2'
        # expected_steps does not need to be exact, approximating is fine.
        logger = sampled_log.SampledWandbLogger(run, expected_steps=10 * 1000 * 1000)
        for i in range(N_STEPS):
            run.log(metrics(100))
        # You MUST call run.finish() explicitly when using SampledWandbLogger
        run.finish()
    """

    def __init__(self, run, expected_steps):
        """Patch a wandb run object, to add SampledLogger behavior.

        Args:
            run: a wandb.Run as returned by wandb.init()
            expected_steps: the number of steps you expect your job to log. This can be
                approximate. It is used to compute the sample rate and target chunk size.
        """
        self._run = run

        # Its fine if this ends up greater than 1, that just means we'll log everything
        self._wandb_log_rate = TARGET_WANDB_LOG_CALLS / expected_steps

        self._history_chunk_steps = expected_steps / MAX_HISTORY_CHUNKS
        if self._history_chunk_steps < MIN_STEPS_PER_HISTORY_CHUNK:
            self._history_chunk_steps = MIN_STEPS_PER_HISTORY_CHUNK

        self._artifact_name = 'run-history-%s' % self._run.id

        self._dirname = os.path.join('history', self._run.id)
        os.makedirs(self._dirname, exist_ok=True)

        self._history_chunk = []

        # monkey patch run. This way we receive the added _step, _runtime, _timestamp
        # fields that wandb.log() and friends add.
        self._real_run_history_cb = run._history_callback
        run.history._callback = self._run_history_callback

        # Currently, the run teardown_hooks are only called on explicit run.finish(), so
        # the user must call run.finish()!
        run._teardown_hooks.insert(0, self._finish)

    def _run_history_callback(self, row=None, step=None):
        assert isinstance(row, dict), "Records passed to log must be dicts"

        if random.random() < self._wandb_log_rate:
            self._real_run_history_cb(row, step)

        self._history_chunk.append(row)
        if len(self._history_chunk) >= self._history_chunk_steps:
            self._flush()

    def _flush(self):
        # synchronous flush to disk
        if not self._history_chunk:
            return
        start_step = self._history_chunk[0]["_step"]
        df = pd.DataFrame(self._history_chunk)
        self._history_chunk = []

        fname = os.path.join(
            self._dirname, 'history-%012d.parquet' % start_step)
        fastparquet.write(fname, df, compression="GZIP")

    def _finish(self):
        self._flush()

        artifact = wandb.Artifact(self._artifact_name, type='history')
        artifact.add_dir(self._dirname)

        self._run.log_artifact(artifact)
