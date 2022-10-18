
"""Prosody Network related modules."""

import logging
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ProsodyEncoder(torch.nn.Module):
    """ Mel-Style Encoder """

    def __init__(self):
        super(ProsodyEncoder, self).__init__()
        # n_position = model_config["max_seq_len"] + 1
        # melencoder:
        # encoder_hidden: 128
        # spectral_layer: 2
        # temporal_layer: 2
        # slf_attn_layer: 1
        # slf_attn_head: 2
        # conv_kernel_size: 5
        # encoder_dropout: 0.1
        # add_llayer_for_adv: True
        n_mel_channels = 256 #model_config["odim"]
        d_melencoder = 256 #model_config["melencoder"]["encoder_hidden"]
        n_spectral_layer = 2 #model_config["melencoder"]["spectral_layer"]
        n_temporal_layer = 2 #model_config["melencoder"]["temporal_layer"]
        n_slf_attn_layer = 1 #model_config["melencoder"]["slf_attn_layer"]
        n_slf_attn_head = 4 #model_config["melencoder"]["slf_attn_head"]
        d_k = d_v = (
            128 #model_config["melencoder"]["encoder_hidden"]
            // 4 #model_config["melencoder"]["slf_attn_head"]
        )
        kernel_size = 5 #model_config["melencoder"]["conv_kernel_size"]
        dropout = 0.2 #model_config["melencoder"]["encoder_dropout"]

        self.add_extra_linear = True #model_config["melencoder"]["add_llayer_for_adv"]

        # self.max_seq_len = model_config["max_seq_len"]

        self.fc_1 = FCBlock(n_mel_channels, d_melencoder)

        self.spectral_stack = torch.nn.ModuleList(
            [
                FCBlock(
                    d_melencoder, d_melencoder, activation=Mish()
                )
                for _ in range(n_spectral_layer)
            ]
        )

        self.temporal_stack = torch.nn.ModuleList(
            [
                nn.Sequential(
                    Conv1DBlock(
                        d_melencoder, 2 * d_melencoder, kernel_size, activation=Mish(), dropout=dropout
                    ),
                    nn.GLU(),
                )
                for _ in range(n_temporal_layer)
            ]
        )

        self.slf_attn_stack = torch.nn.ModuleList(
            [
                MultiHeadAttention(
                    n_slf_attn_head, d_melencoder, d_k, d_v, dropout=dropout, layer_norm=True
                )
                for _ in range(n_slf_attn_layer)
            ]
        )

        self.fc_2 = FCBlock(d_melencoder, d_melencoder)

        if self.add_extra_linear:
            self.fc_3 = FCBlock(d_melencoder, d_melencoder)

    def forward(self, mel, mask):

        max_len = mel.shape[1]
        if mask is not None:
            slf_attn_mask = None#mask.expand(-1, max_len, -1)
        else:
            slf_attn_mask = None

        enc_output = self.fc_1(mel)

        # Spectral Processing
        for _, layer in enumerate(self.spectral_stack):
            enc_output = layer(enc_output)

        # Temporal Processing
        for _, layer in enumerate(self.temporal_stack):
            residual = enc_output
            enc_output = layer(enc_output)
            enc_output = residual + enc_output

        # Multi-head self-attention
        for _, layer in enumerate(self.slf_attn_stack):
            residual = enc_output
            enc_output, _ = layer(
                enc_output, enc_output, enc_output, mask=slf_attn_mask
            )
            enc_output = residual + enc_output

        # Final Layer
        enc_output = self.fc_2(enc_output) # [B, T, H]

        residual = enc_output
        if self.add_extra_linear:
            enc_output = self.fc_3(enc_output)

        # Temporal Average Pooling
        # enc_output = torch.mean(enc_output, dim=1, keepdim=True) # [B, 1, H]

        return enc_output, residual


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class StyleAdaptiveLayerNorm(nn.Module):
    """ Style-Adaptive Layer Norm (SALN) """

    def __init__(self, w_size, hidden_size, bias=False):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(
            w_size,
            2 * hidden_size, # For both b (bias) g (gain) 
            bias,
        )

    def forward(self, h, w):
        """
        h --- [B, T, H_m]
        w --- [B, 1, H_w]
        o --- [B, T, H_m]
        """

        # Normalize Input Features
        mu, sigma = torch.mean(h, dim=-1, keepdim=True), torch.std(h, dim=-1, keepdim=True)
        y = (h - mu) / sigma # [B, T, H_m]

        # Get Bias and Gain
        b, g = torch.split(self.affine_layer(w), self.hidden_size, dim=-1)  # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]

        # Perform Scailing and Shifting
        o = g * y + b # [B, T, H_m]

        return o


class FCBlock(nn.Module):
    """ Fully Connected Block """

    def __init__(self, in_features, out_features, activation=None, bias=False, dropout=None, spectral_norm=False):
        super(FCBlock, self).__init__()
        self.fc_layer = nn.Sequential()
        self.fc_layer.add_module(
            "fc_layer",
            LinearNorm(
                in_features,
                out_features,
                bias,
                spectral_norm,
            ),
        )
        if activation is not None:
            self.fc_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc_layer(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)
        return x


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False, spectral_norm=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
        if spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)

    def forward(self, x):
        x = self.linear(x)
        return x


class Conv1DBlock(nn.Module):
    """ 1D Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, activation=None, dropout=None, spectral_norm=False):
        super(Conv1DBlock, self).__init__()

        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module(
            "conv_layer",
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
                spectral_norm=spectral_norm,
            ),
        )
        if activation is not None:
            self.conv_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x, mask=None):
        x = x.contiguous().transpose(1, 2)
        x = self.conv_layer(x)

        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)

        x = x.contiguous().transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)

        return x


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        spectral_norm=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class SALNFFTBlock(nn.Module):
    """ FFT Block with SALN """

    def __init__(self, d_model, d_w, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(SALNFFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )
        self.layer_norm_1 = StyleAdaptiveLayerNorm(d_w, d_model)
        self.layer_norm_2 = StyleAdaptiveLayerNorm(d_w, d_model)

    def forward(self, enc_input, w, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.layer_norm_1(enc_output, w)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = self.layer_norm_2(enc_output, w)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, layer_norm=False, spectral_norm=False):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = LinearNorm(d_model, n_head * d_k, spectral_norm=spectral_norm)
        self.w_ks = LinearNorm(d_model, n_head * d_k, spectral_norm=spectral_norm)
        self.w_vs = LinearNorm(d_model, n_head * d_v, spectral_norm=spectral_norm)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None

        self.fc = LinearNorm(n_head * d_v, d_model, spectral_norm=spectral_norm)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = output + residual

        return output

# class ProsodyEncoder(torch.nn.Module):
#     """
#     Prosody Encoder Network
#     """
#     def __init__(self, input_size, num_hidden, conv_kernel_size, attention_head, attention_dropout):
#         """
#         :param embedding_size: dimension of embedding
#         :param num_hidden: dimension of hidden
#         """
#         super(ProsodyEncoder, self).__init__()
#         self.spectral_processing_block = torch.nn.Sequential(
#             torch.nn.Linear(input_size, num_hidden),
#             torch.nn.ReLU(),
#             torch.nn.Linear(num_hidden, num_hidden),
#             torch.nn.ReLU(),
#         )

#         self.temporal_processing_unit = torch.nn.Sequential(
#             torch.nn.Conv1d(num_hidden, num_hidden,
#                               kernel_size=conv_kernel_size, stride=1),
#             torch.nn.GLU(dim = -1)
#         )

#         self.attention_block = MultiHeadedAttention(attention_head, num_hidden, attention_dropout)

#         self.final_block = torch.nn.Sequential(
#             torch.nn.Linear(input_size, num_hidden),
#             torch.nn.ReLU()
#         )
    
#     def forward(self, mel, mask):
#         max_len = mel.shape[1]
#         slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

#         prosody_enc_output = self.spectral_processing_block(mel)
