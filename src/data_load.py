import random
import numpy as np
import torch
import pickle 
import os
from collections import OrderedDict
from utils.f0_utils import get_cont_lf0, convert_continuous_f0
import resampy
from .audio_utils import MAX_WAV_VALUE, load_wav, mel_spectrogram, normalize
from .speechsplit_utils import vtlp, get_spenv, get_spmel


def read_fids(fid_list_f):
    with open(fid_list_f, 'r') as f:
        fids = [l.strip().split()[0] for l in f if l.strip()]
    return fids   

class VcDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        meta_file: str, 
        ppg_dir: str,
        f0_dir: str,
        mel_dir: str,
        ppg_file_ext: str = "ppg.npy",
        f0_file_ext: str = "f0.npy",
        mel_file_ext: str = "mel.npy",
    ):
        self.fid_list = read_fids(meta_file)
        self.ppg_dir = ppg_dir
        self.f0_dir = f0_dir
        self.mel_dir = mel_dir
        self.ppg_file_ext = ppg_file_ext
        self.f0_file_ext = f0_file_ext
        self.mel_file_ext = mel_file_ext

        random.seed(1234)
        random.shuffle(self.fid_list)
        print(f'[INFO] Got {len(self.fid_list)} samples.')   
    
    def __getitem__(self, index):
        fid = self.fid_list[index]
        # 1. Load features
        ppg = np.load(f"{self.ppg_dir}/{fid}.{self.ppg_file_ext}")
        if ppg.shape[-1] == 75:
            ppg = ppg[:, 2:-1]  # drop <blank>, <unk> and <sos/eos> dims
        f0 = np.load(f"{self.f0_dir}/{fid}.{self.f0_file_ext}")
        mel = np.load(f"{self.mel_dir}/{fid}.{self.mel_file_ext}")
        
        # 2. Convert f0 to continuous log-f0 and u/v flags
        uv, cont_lf0 = get_cont_lf0(f0, 10.0, False)
        lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
        
        # 3. Convert numpy array to torch.tensor
        ppg = torch.from_numpy(ppg)
        lf0_uv = torch.from_numpy(lf0_uv)
        mel = torch.from_numpy(mel)

        return (ppg, lf0_uv, mel, fid)

    def __len__(self):
        return len(self.fid_list)


class OneshotVcDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file: str,
        vctk_ppg_dir: str,
        libri_ppg_dir: str,
        vctk_f0_dir: str,
        libri_f0_dir: str,
        vctk_wav_dir: str,
        libri_wav_dir: str,
        vctk_spk_dvec_dir: str,
        libri_spk_dvec_dir: str,
        min_max_norm_mel: bool = False,
        mel_min: float = None,
        mel_max: float = None,
        ppg_file_ext: str = "ling_feat.npy",
        f0_file_ext: str = "f0.npy",
        wav_file_ext: str = "wav",
    ):
        self.fid_list = read_fids(meta_file)
        self.vctk_ppg_dir = vctk_ppg_dir
        self.libri_ppg_dir = libri_ppg_dir
        self.vctk_f0_dir = vctk_f0_dir
        self.libri_f0_dir = libri_f0_dir
        self.vctk_wav_dir = vctk_wav_dir
        self.libri_wav_dir = libri_wav_dir
        self.vctk_spk_dvec_dir = vctk_spk_dvec_dir
        self.libri_spk_dvec_dir = libri_spk_dvec_dir

        self.ppg_file_ext = ppg_file_ext
        self.f0_file_ext = f0_file_ext
        self.wav_file_ext = wav_file_ext

        self.min_max_norm_mel = min_max_norm_mel
        if min_max_norm_mel:
            print("[INFO] Min-Max normalize Melspec.")
            assert mel_min is not None
            assert mel_max is not None
            self.mel_max = mel_max
            self.mel_min = mel_min
        
        random.seed(1234)
        random.shuffle(self.fid_list)
        print(f'[INFO] Got {len(self.fid_list)} samples.')
        
    def __len__(self):
        return len(self.fid_list)
    
    def get_spk_dvec(self, fid):
        spk_name = fid.split("_")[0]
        if spk_name.startswith("p"):
            spk_dvec_path = f"{self.vctk_spk_dvec_dir}/{spk_name}.npy"
        else:
            spk_dvec_path = f"{self.libri_spk_dvec_dir}/{spk_name}.npy"
        return torch.from_numpy(np.load(spk_dvec_path))
    
    def compute_mel(self, wav_path):
        audio, sr = load_wav(wav_path)
        if sr != 24000:
            audio = resampy.resample(audio, sr, 24000)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio).unsqueeze(0)
        melspec = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=80,
            sampling_rate=24000,
            hop_size=240,
            win_size=1024,
            fmin=0,
            fmax=8000,
        )
        return melspec.squeeze(0).numpy().T

    def bin_level_min_max_norm(self, melspec):
        # frequency bin level min-max normalization to [-4, 4]
        mel = (melspec - self.mel_min) / (self.mel_max - self.mel_min) * 8.0 - 4.0
        return np.clip(mel, -4., 4.)   

    def __getitem__(self, index):
        fid = self.fid_list[index]
        
        # 1. Load features
        if fid.startswith("p"):
            # vctk
            ppg = np.load(f"{self.vctk_ppg_dir}/{fid}.{self.ppg_file_ext}")
            f0 = np.load(f"{self.vctk_f0_dir}/{fid}.{self.f0_file_ext}")
            mel = self.compute_mel(f"{self.vctk_wav_dir}/{fid}.{self.wav_file_ext}")
        else:
            # libritts
            ppg = np.load(f"{self.libri_ppg_dir}/{fid}.{self.ppg_file_ext}")
            f0 = np.load(f"{self.libri_f0_dir}/{fid}.{self.f0_file_ext}")
            mel = self.compute_mel(f"{self.libri_wav_dir}/{fid}.{self.wav_file_ext}")
        if self.min_max_norm_mel:
            mel = self.bin_level_min_max_norm(mel)
        
        f0, ppg, mel = self._adjust_lengths(f0, ppg, mel)
        spk_dvec = self.get_spk_dvec(fid)

        # 2. Convert f0 to continuous log-f0 and u/v flags
        uv, cont_lf0 = get_cont_lf0(f0, 10.0, False)
        # cont_lf0 = (cont_lf0 - np.amin(cont_lf0)) / (np.amax(cont_lf0) - np.amin(cont_lf0))
        # cont_lf0 = self.utt_mvn(cont_lf0)
        lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)

        # uv, cont_f0 = convert_continuous_f0(f0)
        # cont_f0 = (cont_f0 - np.amin(cont_f0)) / (np.amax(cont_f0) - np.amin(cont_f0))
        # lf0_uv = np.concatenate([cont_f0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
        
        # 3. Convert numpy array to torch.tensor
        ppg = torch.from_numpy(ppg)
        lf0_uv = torch.from_numpy(lf0_uv)
        mel = torch.from_numpy(mel)
        
        return (ppg, lf0_uv, mel, spk_dvec, fid)

    def check_lengths(self, f0, ppg, mel):
        LEN_THRESH = 10
        assert abs(len(ppg) - len(f0)) <= LEN_THRESH, \
            f"{abs(len(ppg) - len(f0))}"
        assert abs(len(mel) - len(f0)) <= LEN_THRESH, \
            f"{abs(len(mel) - len(f0))}"
    
    def _adjust_lengths(self, f0, ppg, mel):
        #self.check_lengths(f0, ppg, mel)
        min_len = min(
            len(f0),
            len(ppg),
            len(mel),
        )
        f0 = f0[:min_len]
        ppg = ppg[:min_len]
        mel = mel[:min_len]
        return f0, ppg, mel


class MultiSpkVcDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file: str,
        ppg_dir: str,
        f0_dir: str,
        mel_dir: str,
        ppg_file_ext: str = "bnf.npy",
        f0_file_ext: str = "f0.npy",
        mel_file_ext: str = "mel.npy",
        scale_mel_to_4: bool = True
    ):
        self.fid_list = read_fids(meta_file)
        self.ppg_dir = ppg_dir
        self.f0_dir = f0_dir
        self.mel_dir = mel_dir
        self.ppg_file_ext = ppg_file_ext
        self.f0_file_ext = f0_file_ext
        self.mel_file_ext = mel_file_ext
        self.scale_mel_to_4 = scale_mel_to_4
        self.spk2idx = self.get_spk2idx()
        random.seed(1234)
        random.shuffle(self.fid_list)
        print(f'[INFO] Got {len(self.fid_list)} samples.')   
        
    def get_spk2idx(self, spk_list_file=None, verbose=False):
        if spk_list_file is None:
            spk_list_file = "/home/shaunxliu/data_96/vctk/spk_set_vctk_arctic.txt"
        with open(spk_list_file, 'r') as f:
            lines = [l.strip() for l in f]
        spk2idx = dict(zip(lines, range(len(lines))))
        if verbose:
            print("Speaker-to-id mapping: \n\t", spk2idx)
        return spk2idx
    
    def __len__(self):
        return len(self.fid_list)

    def utt_mvn(self, cont_lf0):
        cont_lf0 = (cont_lf0 - np.mean(cont_lf0)) / (1e-8 + np.std(cont_lf0))
        return cont_lf0   

    def __getitem__(self, index):
        fid = self.fid_list[index]
        spk_name = fid.split("_")[0]
        spk_id = int(self.spk2idx[spk_name])

        # 1. Load features
        ppg = np.load(f"{self.ppg_dir}/{fid}.{self.ppg_file_ext}")
        if ppg.shape[-1] == 75:
            ppg = ppg[:, 2:-1]  # drop <blank>, <unk> and <sos/eos> dims
        f0 = np.load(f"{self.f0_dir}/{fid}.{self.f0_file_ext}")
        mel = np.load(f"{self.mel_dir}/{fid}.{self.mel_file_ext}")
        
        if len(ppg) < len(f0) // 2:
            if f0.shape[0] < 4 * ppg.shape[0]:
                pads = 4 * ppg.shape[0] - f0.shape[0]
                f0 = np.pad(f0, (0, pads), mode="constant")
            else:
                f0 = f0[:4*ppg.shape[0]]
        else:
            if f0.shape[0] < ppg.shape[0]:
                pads = ppg.shape[0] - f0.shape[0]
                f0 = np.pad(f0, (0, pads), mode="constant")
            else:
                f0 = f0[:ppg.shape[0]]
            
        # 2. Convert f0 to continuous log-f0 and u/v flags
        uv, cont_lf0 = get_cont_lf0(f0, 10.0, False)
        # cont_lf0 = (cont_lf0 - np.amin(cont_lf0)) / (np.amax(cont_lf0) - np.amin(cont_lf0))
        # cont_lf0 = self.utt_mvn(cont_lf0)
        lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)

        # uv, cont_f0 = convert_continuous_f0(f0)
        # cont_f0 = (cont_f0 - np.amin(cont_f0)) / (np.amax(cont_f0) - np.amin(cont_f0))
        # lf0_uv = np.concatenate([cont_f0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
        
        # 3. Convert numpy array to torch.tensor
        ppg = torch.from_numpy(ppg)
        lf0_uv = torch.from_numpy(lf0_uv)
        mel = torch.from_numpy(mel)
        
        if self.scale_mel_to_4:
            mel = mel * 8.0 - 4.0

        return (ppg, lf0_uv, mel, spk_id, fid)


class MultiSpkVcCollate():
    """Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1, give_uttids=False,
                 f02ppg_length_ratio=1, use_spk_dvec=False):
        self.n_frames_per_step = n_frames_per_step
        self.give_uttids = give_uttids
        self.f02ppg_length_ratio = f02ppg_length_ratio
        self.use_spk_dvec = use_spk_dvec

    def __call__(self, batch):
        batch_size = len(batch)              
        # Prepare different features 
        # Input = (ppg, mel, spk_dvec, prosody_vec, fid, lwav)
        ppgs = [x[0] for x in batch]
        #lf0_uvs = [x[1] for x in batch]
        mels = [x[1] for x in batch]
        fids = [x[-2] for x in batch]
        lwav = [x[-1] for x in batch]
        maxlwav = max(lwav)
        # print(f'torch batch length : {batch[0]}')
        spk_ids = [x[2] for x in batch]
        spk_ids = torch.stack(spk_ids).float()

        prosody_vec = [x[3] for x in batch]
        # if len(batch[0]) == 5:
        #     spk_ids = [x[2] for x in batch]
        #     if self.use_spk_dvec:
        #         # use d-vector
        #         spk_ids = torch.stack(spk_ids).float()
        #     else:
        #         # use one-hot ids
        #         spk_ids = torch.LongTensor(spk_ids)
        # Pad features into chunk
        ppg_lengths = [x.shape[0] for x in ppgs]
        mel_lengths = [x.shape[0] for x in mels]
        prosody_vec_lengths = [x.shape[0] for x in prosody_vec]
        max_ppg_len = max(ppg_lengths)
        max_mel_len = max(mel_lengths)
        max_prosody_vec_len = max(prosody_vec_lengths)
        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += (self.n_frames_per_step - max_mel_len % self.n_frames_per_step)
        ppg_dim = ppgs[0].shape[1]
        mel_dim = mels[0].shape[1]
        prosody_vec_dim = prosody_vec[0].shape[1]
        ppgs_padded = torch.FloatTensor(batch_size, max_ppg_len, ppg_dim).zero_()
        mels_padded = torch.FloatTensor(batch_size, max_mel_len, mel_dim).zero_()
        #lf0_uvs_padded = torch.FloatTensor(batch_size, self.f02ppg_length_ratio * max_ppg_len, 2).zero_()
        stop_tokens = torch.FloatTensor(batch_size, max_mel_len).zero_()
        prosody_vec_padded = torch.FloatTensor(batch_size, max_prosody_vec_len, prosody_vec_dim).zero_()
        for i in range(batch_size):
            cur_ppg_len = ppgs[i].shape[0]
            cur_mel_len = mels[i].shape[0]
            ppgs_padded[i, :cur_ppg_len, :] = ppgs[i]
            #lf0_uvs_padded[i, :self.f02ppg_length_ratio*cur_ppg_len, :] = lf0_uvs[i]
            mels_padded[i, :cur_mel_len, :] = mels[i]
            stop_tokens[i, cur_ppg_len-self.n_frames_per_step:] = 1
            cur_prosody_vec_len = prosody_vec[i].shape[0]
            prosody_vec_padded[i, :cur_prosody_vec_len, :] = prosody_vec[i]
        if False:
            print('hererererere')
            ret_tup = (ppgs_padded, mels_padded, torch.LongTensor(ppg_lengths), \
                torch.LongTensor(mel_lengths), spk_ids, stop_tokens,torch.IntTensor(maxlwav))
            if self.give_uttids:
                return ret_tup + (fids, )
            else:
                return ret_tup
        else:
            ret_tup = (ppgs_padded, mels_padded, torch.LongTensor(ppg_lengths), \
                torch.LongTensor(mel_lengths), spk_ids, prosody_vec_padded, \
                torch.LongTensor(prosody_vec_lengths), stop_tokens, torch.LongTensor([maxlwav]))
            if self.give_uttids:
                return ret_tup + (fids, )
            else:
                return ret_tup


class VcCollate():
    """Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1, give_uttids=False):
        self.n_frames_per_step = n_frames_per_step
        self.give_uttids = give_uttids

    def __call__(self, batch):
        batch_size = len(batch)              
        # Prepare different features 
        ppgs = [x[0] for x in batch]
        lf0_uvs = [x[1] for x in batch]
        mels = [x[2] for x in batch]
        fids = [x[-1] for x in batch]
        if len(batch[0]) == 5:
            spk_ids = [x[3] for x in batch]

        # Pad features into chunk
        ppg_lengths = [x.shape[0] for x in ppgs]
        max_ppg_len = max(ppg_lengths)
        if max_ppg_len % self.n_frames_per_step != 0:
            max_ppg_len += (self.n_frames_per_step - max_ppg_len % self.n_frames_per_step)
        ppg_dim = ppgs[0].shape[1]
        mel_dim = mels[0].shape[1]
        ppgs_padded = torch.FloatTensor(batch_size, max_ppg_len, ppg_dim).zero_()
        mels_padded = torch.FloatTensor(batch_size, max_ppg_len, mel_dim).zero_()
        lf0_uvs_padded = torch.FloatTensor(batch_size, max_ppg_len, 2).zero_()
        for i in range(batch_size):
            cur_len = ppgs[i].shape[0]
            ppgs_padded[i, :cur_len, :] = ppgs[i]
            lf0_uvs_padded[i, :cur_len, :] = lf0_uvs[i]
            mels_padded[i, :cur_len, :] = mels[i]
        if self.give_uttids:
            return ppgs_padded, lf0_uvs_padded, mels_padded, torch.LongTensor(ppg_lengths), \
                fids
        else:
            return ppgs_padded, lf0_uvs_padded, mels_padded, torch.LongTensor(ppg_lengths)

class OneshotArciticVcDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file: str,
        arctic_ppg_dir: str,
        arctic_f0_dir: str,
        arctic_wav_dir: str,
        arctic_spk_dvec_dir: str,
        prosody_vec_dir: str,
        min_max_norm_mel: bool = False,
        mel_min: float = None,
        mel_max: float = None,
        ppg_file_ext: str = "ling_feat.npy",
        f0_file_ext: str = "f0.npy",
        wav_file_ext: str = "wav",
        prosody_ext: str = "npy",
    ):
        self.fid_list = read_fids(meta_file)
        self.arctic_ppg_dir = arctic_ppg_dir
        self.arctic_f0_dir = arctic_f0_dir
        self.arctic_wav_dir = arctic_wav_dir
        self.arctic_spk_dvec_dir = arctic_spk_dvec_dir
        self.prosody_vec_dir = prosody_vec_dir


        self.ppg_file_ext = ppg_file_ext
        self.f0_file_ext = f0_file_ext
        self.wav_file_ext = wav_file_ext
        self.prosody_ext = prosody_ext

        self.min_max_norm_mel = min_max_norm_mel
        if min_max_norm_mel:
            print("[INFO] Min-Max normalize Melspec.")
            assert mel_min is not None
            assert mel_max is not None
            self.mel_max = mel_max
            self.mel_min = mel_min
        
        random.seed(1234)
        random.shuffle(self.fid_list)
        print(f'[INFO] Got {len(self.fid_list)} samples.')
        
    def __len__(self):
        return len(self.fid_list)
    
    def get_spk_dvec(self, fid):
        spk_dvec_path = f"{self.arctic_spk_dvec_dir}/{fid}.npy"
        return torch.from_numpy(np.load(spk_dvec_path))
    
    def get_prosody_input(self, fid): #ppg-ERMS-arctic_a0343.npy
        sprf , wfle = fid.split('/')
        prosody_vec = np.load(f"{self.prosody_vec_dir}/ppg-{sprf}-{wfle}.{self.prosody_ext}")
        return prosody_vec

    def get_ppg_input(self, fid): #ppg-ERMS-arctic_a0343.npy
        sprf , wfle = fid.split('/')
        ppg = np.load(f"{self.arctic_ppg_dir}/ppg-{sprf}-{wfle}.{self.ppg_file_ext}")
        return ppg
    
    def compute_mel(self, wav_path):
        audio, sr = load_wav(wav_path)
        lwav = len(audio)
        if sr != 24000:
            audio = resampy.resample(audio, sr, 24000)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio).unsqueeze(0)
        melspec = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=80,
            sampling_rate=24000,
            hop_size=240,
            win_size=1024,
            fmin=0,
            fmax=8000,
        )
        return melspec.squeeze(0).numpy().T, lwav

    def bin_level_min_max_norm(self, melspec):
        # frequency bin level min-max normalization to [-4, 4]
        mel = (melspec - self.mel_min) / (self.mel_max - self.mel_min) * 8.0 - 4.0
        return np.clip(mel, -4., 4.)   

    def __getitem__(self, index):
        fid = self.fid_list[index]
        
        # 1. Load features
        sprf , wfle = fid.split('/')
        ppg = self.get_ppg_input(fid)
        # f0 = np.load(f"{self.arctic_f0_dir}/{fid}.{self.f0_file_ext}")
        mel, lwav = self.compute_mel(f"{self.arctic_wav_dir}/{sprf}/wav/{wfle}.{self.wav_file_ext}")
        if self.min_max_norm_mel:
            mel = self.bin_level_min_max_norm(mel)
        
        prosody_vec = self.get_prosody_input(fid)

        #f0, ppg, mel, prosody_vec = self._adjust_lengths(f0, ppg, mel, prosody_vec)
        spk_dvec = self.get_spk_dvec(fid)

        # 2. Convert f0 to continuous log-f0 and u/v flags
        # uv, cont_lf0 = get_cont_lf0(f0, 10.0, False)
        # cont_lf0 = (cont_lf0 - np.amin(cont_lf0)) / (np.amax(cont_lf0) - np.amin(cont_lf0))
        # cont_lf0 = self.utt_mvn(cont_lf0)
        # lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)

        # uv, cont_f0 = convert_continuous_f0(f0)
        # cont_f0 = (cont_f0 - np.amin(cont_f0)) / (np.amax(cont_f0) - np.amin(cont_f0))
        # lf0_uv = np.concatenate([cont_f0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
        
        # 3. Convert numpy array to torch.tensor
        ppg = torch.from_numpy(ppg)
        # lf0_uv = torch.from_numpy(lf0_uv)
        mel = torch.from_numpy(mel)
        prosody_vec = torch.from_numpy(prosody_vec)
        return (ppg, mel, spk_dvec, prosody_vec, fid, lwav)

    def check_lengths(self, f0, ppg, mel):
        LEN_THRESH = 10
        assert abs(len(ppg) - len(f0)) <= LEN_THRESH, \
            f"{abs(len(ppg) - len(f0))}"
        assert abs(len(mel) - len(f0)) <= LEN_THRESH, \
            f"{abs(len(mel) - len(f0))}"
    
    def _adjust_lengths(self, f0, ppg, mel, prosody_vec):
        #self.check_lengths(f0, ppg, mel)
        min_len = min(
            len(f0),
            len(ppg),
            len(mel),
            len(prosody_vec),
        )
        f0 = f0[:min_len]
        ppg = ppg[:min_len]
        mel = mel[:min_len]
        prosody_vec = prosody_vec[:min_len]
        return f0, ppg, mel, prosody_vec


class Utterances(torch.utils.data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(
        self, 
        meta_file: str,
        feat_dir: str,
        wav_dir: str,
        f0_dir: str,
        org_wav_dir: str,
        spk_dvec_dir: str,
        min_max_norm_mel: bool = False,
        mel_min: float = None,
        mel_max: float = None,
        ):
        """Initialize and preprocess the Utterances dataset."""
        self.fid_list = read_fids(meta_file)

        self.feat_dir = feat_dir
        self.wav_dir = os.path.join(self.feat_dir, wav_dir)
        self.spk_dvec_dir = spk_dvec_dir
        self.f0_dir = os.path.join(self.feat_dir, f0_dir)
        self.org_wav_dir = org_wav_dir
        self.min_max_norm_mel = min_max_norm_mel

        if min_max_norm_mel:
            print("[INFO] Min-Max normalize Melspec.")
            assert mel_min is not None
            assert mel_max is not None
            self.mel_max = mel_max
            self.mel_min = mel_min

        random.seed(1234)
        random.shuffle(self.fid_list)
        print(f'[INFO] Got {len(self.fid_list)} samples.')
        
    def __len__(self):
        return len(self.fid_list)
    
    def compute_mel(self, wav_path):
        audio, sr = load_wav(wav_path)
        lwav = len(audio)
        if sr != 24000:
            audio = resampy.resample(audio, sr, 24000)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio).unsqueeze(0)
        melspec = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=80,
            sampling_rate=24000,
            hop_size=240,
            win_size=1024,
            fmin=0,
            fmax=8000,
        )
        return melspec.squeeze(0).numpy().T, lwav

    def bin_level_min_max_norm(self, melspec):
        # frequency bin level min-max normalization to [-4, 4]
        mel = (melspec - self.mel_min) / (self.mel_max - self.mel_min) * 8.0 - 4.0
        return np.clip(mel, -4., 4.)

    def get_spk_dvec(self, fid):
        spk_dvec_path = f"{self.spk_dvec_dir}/{fid}.npy"
        return torch.from_numpy(np.load(spk_dvec_path))

    def __getitem__(self, index):
        fid = self.fid_list[index]

        wav_mono = np.load(os.path.join(self.wav_dir, f'{fid}.npy'))
        sprf , wfle = fid.split('/')
        spmel_path = os.path.join(self.org_wav_dir, sprf, "wav", wfle.split(".")[0]+".wav")
        f0 = np.load(os.path.join(self.f0_dir, f'{fid}.npy'))

        spk_id_org = None
        emb_org = self.get_spk_dvec(fid)

        spmel, _ = self.compute_mel(spmel_path)
        if self.min_max_norm_mel:
            spmel = self.bin_level_min_max_norm(spmel)
        
        spmel = torch.from_numpy(spmel)

        alpha = np.random.uniform(low=0.9, high=1.1)
        wav_mono = vtlp(wav_mono, 16000, alpha)
        
        spenv = get_spenv(wav_mono)
        spmel_mono = get_spmel(wav_mono)
        rhythm_input = torch.from_numpy(spenv)
        content_input = torch.from_numpy(spmel_mono)
        pitch_input = torch.from_numpy(f0)
        timbre_input = emb_org
        
        return wav_mono, spk_id_org, spmel, rhythm_input, content_input, pitch_input, timbre_input


class SpeechSplitCollate():
    """Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1, give_uttids=False,
                 f02ppg_length_ratio=1, use_spk_dvec=False):
        self.n_frames_per_step = n_frames_per_step
        self.give_uttids = give_uttids
        self.f02ppg_length_ratio = f02ppg_length_ratio
        self.use_spk_dvec = use_spk_dvec

    def __call__(self, batch):
        batch_size = len(batch)              
        # Prepare different features 
        # Input = (wav_mono, spk_id_org, spmel, rhythm_input, content_input, pitch_input, timbre_input)
        # Output = (spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop)
        
        content_input = [x[4] for x in batch]
        pitch_input = [x[5] for x in batch]
        timbre_input = [x[6] for x in batch]
        rhythm_input = [x[3] for x in batch]

        mels = [x[2] for x in batch]

        timbre_input = torch.stack(timbre_input).float()

        ppg_lengths = [int(1.2*x.shape[0]) for x in content_input]
        mel_lengths = [x.shape[0] for x in mels]

        max_ppg_len = max(ppg_lengths)
        max_mel_len = max(mel_lengths)

        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += (self.n_frames_per_step - max_mel_len % self.n_frames_per_step)

        content_dim = content_input[0].shape[1]
        #print(f'pitch len : {len(pitch_input)} shape: {pitch_input[0].shape}')
        #pitch_dim = pitch_input[0].shape[1]
        rhythm_dim = rhythm_input[0].shape[1]
        mel_dim = mels[0].shape[1]

        content_padded = torch.FloatTensor(batch_size, max_ppg_len, content_dim).zero_()
        pitch_padded = torch.FloatTensor(batch_size, max_ppg_len).zero_()
        rhythm_padded = torch.FloatTensor(batch_size, max_ppg_len, rhythm_dim).zero_()
        mels_padded = torch.FloatTensor(batch_size, max_mel_len, mel_dim).zero_()
        stop_tokens = torch.FloatTensor(batch_size, max_mel_len).zero_()

        for i in range(batch_size):
            curr_content_len = content_input[i].shape[0]
            curr_pitch_len = pitch_input[i].shape[0]
            curr_rhythm_len = rhythm_input[i].shape[0]
            curr_mel_len = mels[i].shape[0]

            content_padded[i, :curr_content_len, :] = content_input[i]
            pitch_padded[i, :curr_pitch_len] = pitch_input[i]
            rhythm_padded[i, :curr_rhythm_len, :] = rhythm_input[i]
            mels_padded[i, :curr_mel_len, :] = mels[i]

            stop_tokens[i, curr_content_len-self.n_frames_per_step:] = 1
        
        ret_tup = (content_padded, pitch_padded, rhythm_padded, mels_padded, \
            torch.LongTensor(ppg_lengths), torch.LongTensor(mel_lengths), timbre_input, stop_tokens)
        
        return ret_tup
    

class Collator(object):
    def __init__(self, config):
        self.min_len_seq = config.min_len_seq
        self.max_len_seq = config.max_len_seq
        self.max_len_pad = config.max_len_pad

    def __call__(self, batch):
        new_batch = []
        for token in batch:

            _, spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input = token
            len_crop = np.random.randint(self.min_len_seq, self.max_len_seq+1)
            left = np.random.randint(0, len(spmel_gt)-len_crop)

            spmel_gt = spmel_gt[left:left+len_crop, :] # [Lc, F]
            rhythm_input = rhythm_input[left:left+len_crop, :] # [Lc, F]
            content_input = content_input[left:left+len_crop, :] # [Lc, F]
            pitch_input = pitch_input[left:left+len_crop] # [Lc, ]
            
            spmel_gt = np.clip(spmel_gt, 0, 1)
            rhythm_input = np.clip(rhythm_input, 0, 1)
            content_input = np.clip(content_input, 0, 1)
            
            spmel_gt = np.pad(spmel_gt, ((0,self.max_len_pad-spmel_gt.shape[0]),(0,0)), 'constant')
            rhythm_input = np.pad(rhythm_input, ((0,self.max_len_pad-rhythm_input.shape[0]),(0,0)), 'constant')
            content_input = np.pad(content_input, ((0,self.max_len_pad-content_input.shape[0]),(0,0)), 'constant')
            pitch_input = np.pad(pitch_input[:,np.newaxis], ((0,self.max_len_pad-pitch_input.shape[0]),(0,0)), 'constant', constant_values=-1e10)
            
            new_batch.append( (spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop) ) 
            
        batch = new_batch  
        spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop = zip(*batch)
        spk_id_org = list(spk_id_org)
        spmel_gt = torch.FloatTensor(np.stack(spmel_gt, axis=0))
        rhythm_input = torch.FloatTensor(np.stack(rhythm_input, axis=0))
        content_input = torch.FloatTensor(np.stack(content_input, axis=0))
        pitch_input = torch.FloatTensor(np.stack(pitch_input, axis=0))
        timbre_input = torch.FloatTensor(np.stack(timbre_input, axis=0))
        len_crop = torch.LongTensor(np.stack(len_crop, axis=0))
        
        return spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop

    
class MultiSampler(torch.utils.data.sampler.Sampler):
    """Samples elements more than once in a single pass through the data.
    """
    def __init__(self, num_samples, n_repeats, shuffle=False):
        self.num_samples = num_samples
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def gen_sample_array(self):
        self.sample_idx_array = torch.arange(self.num_samples, dtype=torch.int64).repeat(self.n_repeats)
        if self.shuffle:
            self.sample_idx_array = self.sample_idx_array[torch.randperm(len(self.sample_idx_array))]
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.sample_idx_array)        


def get_loader(config):
    """Build and return a data loader list."""

    dataset = Utterances(config)
    collator = Collator(config)
    sampler = MultiSampler(len(dataset), config.samplier, shuffle=config.shuffle)
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 sampler=sampler,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True,
                                 worker_init_fn=worker_init_fn,
                                 collate_fn=collator)

    return data_loader