import torch
from torch.utils.data import DataLoader
import numpy as np
from src.solver import BaseSolver
# from src.data_load import VcDataset, VcCollate
from src.data_load import OneshotVcDataset, MultiSpkVcCollate, OneshotArciticVcDataset
from src.rnn_ppg2mel import BiRnnPpg2MelModel
from src.optim import Optimizer
from src.util import human_format, feat_to_fig
from src.loss_fn import MaskedMSELoss


class Solver(BaseSolver):
    """Customized Solver."""
    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        self.best_loss = np.inf

    def fetch_data(self, data):
        """Move data to device"""
        data = [i.to(self.device) for i in data]
        return data

    def load_data(self):
        """ Load data for training/validation/plotting."""
        train_dataset = OneshotArciticVcDataset(
            meta_file=self.config.data.train_fid_list,
            arctic_ppg_dir = self.config.data.arctic_ppg_dir,
            arctic_f0_dir = self.config.data.arctic_f0_dir,
            arctic_wav_dir = self.config.data.arctic_wav_dir,
            arctic_spk_dvec_dir = self.config.data.arctic_spk_dvec_dir,
            ppg_file_ext=self.config.data.ppg_file_ext,
            min_max_norm_mel=self.config.data.min_max_norm_mel,
            mel_min=self.config.data.mel_min,
            mel_max=self.config.data.mel_max,
        )
        dev_dataset = OneshotArciticVcDataset(
            meta_file=self.config.data.dev_fid_list,
            arctic_ppg_dir = self.config.data.arctic_ppg_dir,
            arctic_f0_dir = self.config.data.arctic_f0_dir,
            arctic_wav_dir = self.config.data.arctic_wav_dir,
            arctic_spk_dvec_dir = self.config.data.arctic_spk_dvec_dir,
            ppg_file_ext=self.config.data.ppg_file_ext,
            min_max_norm_mel=self.config.data.min_max_norm_mel,
            mel_min=self.config.data.mel_min,
            mel_max=self.config.data.mel_max,
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            num_workers=self.paras.njobs,
            shuffle=True,
            batch_size=self.config.hparas.batch_size,
            pin_memory=False,
            drop_last=True,
            collate_fn=MultiSpkVcCollate(n_frames_per_step=1,
                                         f02ppg_length_ratio=1,
                                         use_spk_dvec=True),
        )
        self.dev_dataloader = DataLoader(
            dev_dataset,
            num_workers=self.paras.njobs,
            shuffle=False,
            batch_size=self.config.hparas.batch_size,
            pin_memory=False,
            drop_last=False,
            collate_fn=MultiSpkVcCollate(n_frames_per_step=1,
                                         f02ppg_length_ratio=1,
                                         use_spk_dvec=True),
        )
        msg = "Have prepared training set and dev set."
        self.verbose(msg)
    
    def load_pretrained_params(self):
        prefix = "ppg2mel_model"
        ignore_layers = ["ppg2mel_model.spk_embedding.weight"]
        pretrain_model_file = self.config.data.pretrain_model_file
        pretrain_ckpt = torch.load(
            pretrain_model_file, map_location=self.device
        )
        model_dict = self.model.state_dict()
        
        # 1. filter out unnecessrary keys
        pretrain_dict = {k.split(".", maxsplit=1)[1]: v 
                         for k, v in pretrain_ckpt.items() if "spk_embedding" not in k 
                            and "wav2ppg_model" not in k and "reduce_proj" not in k}
        # assert len(pretrain_dict.keys()) == len(model_dict.keys())

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrain_dict)

        # 3. load the new state dict
        self.model.load_state_dict(model_dict)

    def set_model(self):
        """Setup model and optimizer"""
        # Model
        self.model = BiRnnPpg2MelModel(**self.config["model"]).to(self.device)
        if "pretrain_model_file" in self.config.data:
            self.load_pretrained_params()

        # model_params = [{'params': self.model.spk_embedding.weight}]
        model_params = [{'params': self.model.parameters()}]
        
        # Loss criterion
        self.loss_criterion = MaskedMSELoss()

        # Optimizer
        self.optimizer = Optimizer(model_params, **self.config["hparas"])
        self.verbose(self.optimizer.create_msg())

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()

    def exec(self):
        self.verbose("Total training steps {}.".format(
            human_format(self.max_step)))

        mel_loss = None
        n_epochs = 0
        # Set as current time
        self.timer.set()
        
        while self.step < self.max_step:
            for data in self.train_dataloader:
                # Pre-step: updata lr_rate and do zero_grad
                lr_rate = self.optimizer.pre_step(self.step)
                total_loss = 0
                # data to device
                ppgs, mels, in_lengths, \
                    out_lengths, spk_ids, _, _ = self.fetch_data(data)
                self.timer.cnt("rd")
                mel_pred = self.model(
                    ppg=ppgs,
                    ppg_lengths=out_lengths,
                    spembs=spk_ids,
                ) 
                loss = self.loss_criterion(mel_pred, mels, out_lengths)

                self.timer.cnt("fw")

                # Back-prop
                grad_norm = self.backward(loss)
                self.step += 1

                # Logger
                if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
                    self.progress("Tr stat | Loss - {:.4f} | Grad. Norm - {:.2f} | {}"
                                  .format(loss.cpu().item(), grad_norm, self.timer.show()))
                    self.write_log('loss', {'tr': loss})

                # Validation
                if (self.step == 1) or (self.step % self.valid_step == 0):
                    self.validate()

                # End of step
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()
                self.timer.set()
                if self.step > self.max_step:
                    break
            n_epochs += 1
        self.log.close()

    def validate(self):
        self.model.eval()
        dev_loss = 0.0

        for i, data in enumerate(self.dev_dataloader):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.dev_dataloader)))
            # Fetch data
            # ppgs, lf0_uvs, mels, lengths = self.fetch_data(data)
            ppgs, mels, in_lengths, \
                out_lengths, spk_ids, _,_ = self.fetch_data(data)

            with torch.no_grad():
                mel_pred = self.model(
                    ppg=ppgs,
                    ppg_lengths=out_lengths,
                    spembs=spk_ids,
                ) 
                loss = self.loss_criterion(mel_pred, mels, out_lengths)
                dev_loss += loss.cpu().item()

        dev_loss = dev_loss / (i + 1)
        self.save_checkpoint(f'step_{self.step}.pth', 'loss', dev_loss, show_msg=False)
        if dev_loss < self.best_loss:
            self.best_loss = dev_loss
            self.save_checkpoint(f'best_loss_step_{self.step}.pth', 'loss', dev_loss)
        self.write_log('loss', {'dv_loss': dev_loss})

        # Resume training
        self.model.train()

