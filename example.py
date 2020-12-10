import random
import sys
import wandb

import sampled_log

N_STEPS = 30 * 1000 * 1000
METRICS_PER_STEP = 100


def metrics(n):
    m = {}
    for i in range(n):
        m['metric-%s' % i] = random.random()
    return m


def main(argv):
    run = wandb.init(project='sampled-log2')

    # Pass the run object to SampledWandbLogger, along with the
    # expected number of steps for your job.
    #
    # expected_steps does not need to be exact.
    logger = sampled_log.SampledWandbLogger(run, expected_steps=N_STEPS)

    # then run your job as normal, logging metrics by calling run.log
    # etc.
    for i in range(N_STEPS):
        run.log(metrics(METRICS_PER_STEP))

    # You must call run.finish when using SampledWandbLogger (or
    # use the run context manager: `with wandb.init() as run:`)
    run.finish()


if __name__ == '__main__':
    main(sys.argv)
