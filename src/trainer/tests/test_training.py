import torch
import numpy as np

from ..trainer import Trainer


class TestTrainer(Trainer):

    def run_training(self):
        self.on_training_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            with torch.no_grad():
                self._on_validation_epoch_start()
                validation_outputs = []
                plot_on_iter = np.random.choice(self.config.val_iters_per_epoch, size = 2)
                for batch_id in range(self.config.val_iters_per_epoch):
                    validation_outputs.append(self.validation_step(next(self.dataloader_val), plot_batch=(batch_id in plot_on_iter)))
                self._on_validation_epoch_end(validation_outputs)
