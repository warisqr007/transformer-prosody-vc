import os
import torch
import numpy as np
from data_objects.kaldi_interface import KaldiInterface
from src import build_model
from pathlib import Path
from utils.load_yaml import HpsYaml
import resampy
from src.audio_utils import MAX_WAV_VALUE, load_wav, mel_spectrogram, normalize

def compute_mel(wav_path):
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

def build_transf_model(model_config, model_file, device):
    model_class = build_model(model_config["model_name"])
    ppg2mel_model = model_class(
        model_config["model"]
    ).to(device)
    ckpt = torch.load(model_file, map_location=device)
    ppg2mel_model.load_state_dict(ckpt["model"])
    ppg2mel_model.eval()
    return ppg2mel_model


def get_bnfs(spk_id, utterance_id, kaldi_dir):
    ki = KaldiInterface(wav_scp=str(os.path.join(kaldi_dir, 'wav.scp')),
                        bnf_scp=str(os.path.join(kaldi_dir, 'bnf/feats.scp')))
    bnf = ki.get_feature('_'.join([spk_id, utterance_id]), 'bnf')
    return bnf

@torch.no_grad()
def convert(src_speaker_fpath, tgt_prosody_fpath, utterance_id, output_dir):
    
    # Compute prosody representation
    prosody_speaker = os.path.basename(tgt_prosody_fpath)
    # prosody_speaker_kaldi_dir = os.path.join(tgt_prosody_fpath, 'kaldi')
    # prosody_vec = get_bnfs(prosody_speaker, utterance_id, prosody_speaker_kaldi_dir)
    prosody_wav_path = f"{tgt_prosody_fpath}/wav/{utterance_id}.wav"
    prosody_vec, _ = compute_mel(prosody_wav_path)
    prosody_vec = torch.from_numpy(prosody_vec).unsqueeze(0).to(device)
    
    #Compute PPG
    src_speaker = os.path.basename(src_speaker_fpath)
    src_speaker_kaldi_dir = os.path.join(src_speaker_fpath, 'kaldi')    
    ppg = get_bnfs(src_speaker, utterance_id, src_speaker_kaldi_dir)
    ppg = torch.from_numpy(ppg).unsqueeze(0).to(device)

    bnf_pred, att_ws = ppg2ppg_model.inference(torch.squeeze(ppg), torch.squeeze(prosody_vec))
    bnf_pred_npy = bnf_pred.cpu().numpy()


    step = os.path.basename(ppg2ppg_model_file)[:-4].split("_")[-1]
    output_dir = os.path.join(output_dir, 'Step_'+step, prosody_speaker)
    os.makedirs(output_dir, exist_ok=True)

    bnf_fname = f"{output_dir}/{utterance_id}.npy"
    np.save(bnf_fname, bnf_pred_npy, allow_pickle=False)

if __name__ == "__main__":

    ppg2ppg_model_train_config = Path('/mnt/data1/waris/repo/transformer-prosody-vc/conf/transformer_prosody_predictor.yaml')
    ppg2ppg_config = HpsYaml(ppg2ppg_model_train_config) 
    ppg2ppg_model_file = Path('/mnt/data1/waris/repo/transformer-prosody-vc/ckpt/prosody-predictor-mel/best_loss_step_170000.pth')
    device = 'cuda'
    ppg2ppg_model = build_transf_model(ppg2ppg_config, ppg2ppg_model_file, device)

    speakers = ['BDL', 'NJS', 'TXHC', 'YKWK', 'ZHAA']
    #speakers = ['MBMPS', 'BWC', 'HKK', 'SKA']
    utterance_ids = ['arctic_b0534', 'arctic_b0537', 'arctic_b0538', 'arctic_b0539']
    #utterance_ids = ['arctic_a00'+str(i) for i in range(10, 30)] + ['arctic_b0534', 'arctic_b0537', 'arctic_b0538', 'arctic_b0539']

    basepath = '/mnt/data1/waris/datasets/data/arctic_dataset/test_speakers_16k'
    #basepath = '/mnt/data1/waris/datasets/data/arctic_dataset/all_data_for_ac_vc_train'
    output_dir = '/mnt/data1/waris/repo/transformer-prosody-vc/synthesis_output/prosody_corrected_bnfs/prosody_32d_mel_l1_align/'

    for speaker in speakers:
        src_speaker_fpath = os.path.join(basepath, 'BDL')
        tgt_speaker_fpath = os.path.join(basepath, speaker)

        for utterance_id in utterance_ids:
            if speaker=='SKA' and (utterance_id=='arctic_b0537' or utterance_id=='arctic_b0539'):
                continue
            convert(src_speaker_fpath, tgt_speaker_fpath, utterance_id, output_dir)

    #output_dir = '/mnt/data1/waris/repo/transformer-prosody-vc/synthesis_output/prosody_corrected_bnfs/prosody-attn/'



