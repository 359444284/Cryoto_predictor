import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, Destationary_Encoder, \
    Destationaty_EncoderLayer, Destationary_DecoderLayer, Destationary_Decoder
from .layers.SelfAttention_Family import FullAttention, AttentionLayer, Destationary_AttentionLayer, \
    Destationary_Attention
from .layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np
from .BasicModule import BasicModule


class TransformerEnDe(BasicModule):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self):
        super(TransformerEnDe, self).__init__()
        self.model_name = 'transformerende'
        configs = self.config
        self.pred_len = configs.forecast_horizon
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )

        self.encoder = Destationary_Encoder(
            [
                Destationaty_EncoderLayer(
                    Destationary_AttentionLayer(
                        Destationary_Attention(True, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=True),
                        configs.d_model,
                        configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_se=True
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(
        #                 FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation,
        #         )
        #         for l in range(configs.d_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model),
        #     projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        # )

        self.decoder = Destationary_Decoder(
            [
                Destationary_DecoderLayer(
                    Destationary_AttentionLayer(
                        Destationary_Attention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    Destationary_AttentionLayer(
                        Destationary_Attention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.tao_projection = nn.Linear(configs.enc_in * (self.config.lockback_window + 1), 1)
        self.delta_projection = nn.Linear(configs.enc_in * (self.config.lockback_window + 1),
                                          self.config.lockback_window)

        self.enc_mask = None
        self.dec_mask = None

    def forward(self, x_enc, x_mark_enc, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        B, L, C = x_enc.shape
        mean = torch.mean(x_enc, dim=1, keepdim=True)
        std = torch.std(x_enc, dim=1, keepdim=True)
        norm_src = (x_enc - mean) / (std + 1e-8)
        # norm_src = x_enc
        std_cat, mean_cat = torch.flatten(torch.cat([x_enc, std], dim=1), start_dim=1), \
                            torch.flatten(torch.cat([x_enc, mean], dim=1), start_dim=1)
        tao, delta = torch.exp(self.tao_projection(std_cat)), \
                     self.delta_projection(mean_cat)

        delta_en = torch.repeat_interleave(delta.unsqueeze(dim=1), L, dim=1).unsqueeze(dim=1)
        delta_dn = torch.repeat_interleave(delta.unsqueeze(dim=1), (L//2 + self.config.forecast_horizon), dim=1).unsqueeze(dim=1)

        if self.enc_mask is None:
            en_mask = self._generate_square_subsequent_mask(L, L).to(self.device)
            de_mask = self._generate_square_subsequent_mask((L//2 + self.config.forecast_horizon), (L//2 + self.config.forecast_horizon)).to(self.device)
            self.enc_mask = en_mask
            self.dec_mask = de_mask


        enc_out = self.enc_embedding(norm_src, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=self.enc_mask)
        enc_out, attns = self.encoder(enc_out, tao, delta_en, attn_mask=self.enc_mask)

        norm_dec = torch.cat(
            (norm_src[:, L//2:, :], torch.zeros((B, self.config.forecast_horizon, C)).to(self.device)),
            dim=1)
        dec_out = self.dec_embedding(norm_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=self.dec_mask, cross_mask=self.dec_mask)
        dec_out = self.decoder(dec_out, enc_out, tao, delta_dn, x_mask=self.dec_mask, cross_mask=self.dec_mask)

        # out = dec_out[:, -self.pred_len:, -1]
        out = std[:, :, -1] * dec_out[:, -self.pred_len:, -1] + mean[:, :, -1]
        if self.output_attention:
            return out, attns
        else:
            return out  # [B, L, D]

    def _generate_square_subsequent_mask(self, x, y):
        # mask_shape = [B, 1, sz, sz]
        mask = (torch.triu(torch.ones(x, y)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask