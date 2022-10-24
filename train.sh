#!/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=1

########## Train BiLSTM oneshot VC model ##########
# python main.py --config ./conf/bilstm_ppg2mel_vctk_libri_oneshotvc.yaml \
#                --oneshotvc \
#                --bilstm
###################################################

########## Train Seq2seq oneshot VC model ###########
# python main.py --config ./conf/seq2seq_mol_ppg2mel_vctk_libri_oneshotvc.yaml \
#                --oneshotvc \
###################################################

########## Train Own oneshot VC model ##########
# python main.py --config ./conf/own.yaml \
#                --ownarc \
#                --bilstm

########## Train Transformer VC model ##########

# Config 1:
# Speaker Embed = atinput
# pitch = no
# prososy embedding = yes
# python main.py --config /mnt/data1/waris/repo/transformer-vc-prosody/conf/transformer_vc_ppg2mel_outspkdloss_inp_conct.yaml \
#                --name=transformer-vc \
#                --seed=2 \
#                --transvcsplinpconc
#
# Status: Trained
# Results:
#

# Config 1:
# Speaker Embed = atinput
# pitch = no
# prososy embedding = yes
# RR block = yes
# python main.py  --config /mnt/data1/waris/repo/transformer-vc-prosody/conf/transformer_vc_ppg2mel_outspkdloss_inp_conct_with_rr.yaml \
#                 --name=transformer-vc-rr \
#                 --seed=2 \
#                 --transvcsplinpconc
#
# Status: Trained
# Results:
#

# Config 2:
# Speaker Embed = atinput
# pitch = no
# prososy embedding = yes
#PPG AND PROSODY VEC = BNF
python main.py  --config /mnt/data1/waris/repo/transformer-prosody-vc/conf/transformer_vc_ppg2mel_outspkdloss_inp_conct_with_rr.yaml \
                --name=transformer-vc-init-II \
                --seed=2 \
                --transvcsplinpconc

# Status: Trained
# Results:
#

# Config 3:
# Speaker Embed = atinput
# pitch = no
# prososy embedding = yes
# RR block = yes
# query = HS, K/V = prosody
# python main.py  --config /mnt/data1/waris/repo/transformer-prosody-vc/conf/transformer_vc_ppg2mel_outspkdloss_inp_conct_with_rr.yaml \
#                 --name=transformer-vc-rrIV \
#                 --seed=2 \
#                 --transvcsplinpconc
# # 
# Status: Trained
# Results:
#


# Config 3:
# Speaker Embed = atinput
# pitch = yes
# python main.py --config /mnt/data1/waris/repo/transformer-vc/conf/transformer_vc_ppg2mel_outspkdloss_inp_conct_with_pitch.yaml \
#                --name=transformer-vc-config2 \
#                --seed=2 \
#                --transvcsplinpconc
#
# Status: Waiting to run
#

# Config 4:
# Speaker Embed = after_encoder
# pitch = yes
# python main.py --config /mnt/data1/waris/repo/transformer-vc/conf/transformer_vc_ppg2mel_outspkdloss_attn_conct_with_pitch.yaml \
#                --name=transformer-vc-config3 \
#                --seed=2 \
#                --transvcsplinpconc
#
# Status: Waiting to run
#