import os
import queue
import random

import fastparquet
import pandas as pd

import wandb


class SampledWandbLogger(object):
    def __init__(self, run, wandb_log_rate=0.1, history_chunk_size=10):
        self._run = run
        self._dirname = os.path.join('history', self._run.id)
        self._artifact_name = 'run-history-%s' % self._run.name

        os.makedirs(self._dirname, exist_ok=True)

        self._wandb_log_rate = wandb_log_rate
        self._history_chunk_size = history_chunk_size
        self._parq_flush_q = queue.Queue()
        self._parq_seq_num = 0

        # monkey patch run :(
        self._real_run_history_cb = run._history_callback

        print("WANDB RUN", run, run._history_callback)
        run.history._callback = self._run_history_callback

    def _run_history_callback(self, row=None, step=None):
        print('IN PATCHED CB')
        assert isinstance(row, dict), "Records passed to log must be dicts"

        self._real_run_history_cb(row, step)

        self._parq_flush_q.put(row)
        if random.random() < self._wandb_log_rate:
            wandb.log(row)

        if self._parq_flush_q.qsize() >= self._history_chunk_size:
            self.flush()

    def flush(self):
        records = []
        while True:
            try:
                record = self._parq_flush_q.get(block=False)
                records.append(record)
            except queue.Empty:
                break

        if not records:
            return

        records = [r for r in records if "_step" in r]
        df = pd.DataFrame(records, index=[r["_step"] for r in records])
        fname = os.path.join(
            self._dirname, 'history-%s.parquet' % self._parq_seq_num)
        fastparquet.write(fname, df, compression="GZIP")

        artifact = wandb.Artifact(self._artifact_name, type='history')
        artifact.add_dir(self._dirname)

        self._run.log_artifact(artifact)

        self._parq_seq_num += 1
