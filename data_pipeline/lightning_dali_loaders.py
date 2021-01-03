# Dali dataloaders for pytorch lightning
from pytorch_lightning import LightningDataModule
from data_pipeline.basic_pipeline import HybridTrainPipe, HybridValPipe
import torch
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

class PLDaliPipe(LightningDataModule):
    def __init__(self, hparams, train_datadir, val_datadir):
        super().__init__()

        self.hparams = hparams
        self.train_datadir = train_datadir
        self.val_datadir = val_datadir

    def setup(self, stage=None):
        # batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
        #                    data_dir=traindir, crop=crop_size, dali_cpu=False
        device_id = self.trainer.local_rank
        shard_id = self.trainer.global_rank
        num_shards = self.trainer.world_size
        train_pipeline = HybridTrainPipe(batch_size=self.hparams.batch_size, num_threads=8, 
                                            device_id=device_id, data_dir=self.train_datadir, crop=224, 
                                            dali_cpu=False)

        val_pipline = HybridValPipe(batch_size=self.hparams.batch_size, num_threads=8, 
                                            device_id=device_id, data_dir=self.val_datadir, crop=224, size=256)

        class LightningWrapper(DALIClassificationIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                # DDP is used so only one pipeline per process
                # also we need to transform dict returned by DALIClassificationIterator to iterable
                # and squeeze the lables
                out = out[0]
                return [out[k] if k != "label" else torch.squeeze(out[k]) for k in self.output_map]

        self.train_loader = LightningWrapper(train_pipeline, reader_name="Reader", 
                                                auto_reset=True)

        self.val_loader = LightningWrapper(val_pipline, reader_name="Reader", 
                                                auto_reset=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


