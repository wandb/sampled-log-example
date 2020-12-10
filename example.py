import random
import wandb

import sampled_log

run = wandb.init(project='sampled-log')

logger = sampled_log.SampledWandbLogger(run, wandb_log_rate=0.2)

for i in range(100):
    run.log({'a': random.random()})
