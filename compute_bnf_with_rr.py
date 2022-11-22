"""
Compute CTC-Attention Seq2seq ASR encoder bottle-neck features (BNF).
"""
import sys
import os
import argparse
import glob2
import soundfile
import librosa
from joblib import Parallel, delayed
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class InterpLnr(torch.nn.Module):
    
    def __init__(self):
        super().__init__()        
        
    def pad_sequences(self, sequences):
        channel_dim = sequences[0].size()[-1]
        out_dims = (len(sequences), self.max_len_pad, channel_dim)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[:self.max_len_pad]
            
        return out_tensor 
    

    def forward(self, x):

        device = x.device
        batch_size = x.size(0)

        #self.max_len_seq = 1#config.max_len_seq
        self.max_len_seq = x.size(1)
        self.max_len_pad = self.max_len_seq #192#config.max_len_pad
        
        self.min_len_seg = 19 #config.min_len_seg
        self.max_len_seg = 32 #config.max_len_seg
        
        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1
        

        #print(x.device)
        #len_seq = torch.tensor([x.size(1)]).to(x.device)
        len_seq = torch.tensor(self.max_len_pad).expand(batch_size).to(device)
        #print(len_seq)

        
        
        # indices of each sub segment
        indices = torch.arange(self.max_len_seg*2, device=device)\
                  .unsqueeze(0).expand(batch_size*self.max_num_seg, -1)
        # scales of each sub segment
        scales = torch.rand(batch_size*self.max_num_seg, 
                            device=device) + 0.5
        
        idx_scaled = indices / scales.unsqueeze(-1)
        idx_scaled_fl = torch.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        
        len_seg = torch.randint(low=self.min_len_seg, 
                                high=self.max_len_seg, 
                                size=(batch_size*self.max_num_seg,1),
                                device=device)
        
        # end point of each segment
        idx_mask = idx_scaled_fl < (len_seg - 1)
       
        offset = len_seg.view(batch_size, -1).cumsum(dim=-1)
        # offset starts from the 2nd segment
        offset = F.pad(offset[:, :-1], (1,0), value=0).view(-1, 1)
        
        idx_scaled_org = idx_scaled_fl + offset
        
        len_seq_rp = torch.repeat_interleave(len_seq, self.max_num_seg)

        # print(idx_scaled_org.device)
        # print(len_seq_rp.device)
        # print(len_seq.device)
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1)
        
        idx_mask_final = idx_mask & idx_mask_org
        
        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)
        
        index_1 = torch.repeat_interleave(torch.arange(batch_size, 
                                            device=device), counts)
        
        index_2_fl = idx_scaled_org[idx_mask_final].long()
        index_2_cl = index_2_fl + 1
        
        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)
        
        y = (1-lambda_f)*y_fl + lambda_f*y_cl
        
        sequences = torch.split(y, counts.tolist(), dim=0)
       
        seq_padded = self.pad_sequences(sequences)

        #print(seq_padded.size())
        return seq_padded


def compute_bnf_with_rr(
    output_dir: str,
    ppg_file: str
):
    device = "cuda:0"
    
    interplnr = InterpLnr()
    interplnr.to(torch.device("cuda:0"))

    ppg = np.load(ppg_file)
    ppg = torch.from_numpy(ppg).to(device).unsqueeze(0)

    bnf = interplnr(ppg)
    bnf_npy = bnf.squeeze(0).cpu().numpy()

    fid = os.path.basename(ppg_file)
    os.makedirs(output_dir, exist_ok=True)
    bnf_fname = f"{output_dir}/{fid}"
    np.save(bnf_fname, bnf_npy, allow_pickle=False)

def get_parser():
    parser = argparse.ArgumentParser(description="compute bnf with RR")

    parser.add_argument(
        "--output_dir",
        type=str,
        #required=True,
        default="/mnt/data1/waris/model_preprocessing/transformer-prosody-vc/bnfs",
    )
    parser.add_argument(
        "--ppg_dir",
        type=str,
        default="/mnt/data1/waris/datasets/data/arctic_dataset/all_data_for_ac_vc_train/SV2TTS/synthesizer/ppgs",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    kwargs = vars(args)

    ppg_file_list = glob2.glob(f"{args.ppg_dir}/*.npy")
    print(f"Globbing {len(ppg_file_list)} PPG files.")
    Parallel(n_jobs=9)(delayed(compute_bnf_with_rr)(args.output_dir, ppg_file) for ppg_file in tqdm(ppg_file_list))

