# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        # 300
        # 512
        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        # 全局特征融合
        self.lstm_global = nn.LSTM(
            input_size=512,
            hidden_size=1024,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        self.gate = nn.Linear(__C.HIDDEN_SIZE*6, __C.HIDDEN_SIZE*2)

    def forward(self, img_feat, ques_ix):
        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))  # (64,1,1,14)
        img_feat_mask = self.make_mask(img_feat)  # (64,1,1,100)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)  # (64,14,300)
        lang_feat, _ = self.lstm(lang_feat)  # (64,14,512)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat) # (64,100,512)
        global_img_feat = torch.sum(img_feat, dim=-2).unsqueeze(1)/img_feat.shape[1] # (64,1,512)
        global_img_feat_mask = self.make_mask(global_img_feat) # (64, 1,1,1)
        img_feat = torch.cat((img_feat, global_img_feat), -2) # (64,101,512)
        img_feat_mask = torch.cat((img_feat_mask, global_img_feat_mask), -1)

        # Backbone Framework
        # 这里面是注意力机制，我们要改的也是这里
        lang_feat, img_feat, global_all_feature = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        ) # (64,1024)

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask[:, :, :, 0:100]
        ) # (64,1024)

        # 全局图像特征进行LSTM进行融合
        global_all_feature, _ = self.lstm_global(global_all_feature)
        # print(global_all_feature1.shape)  # (64,6,1024)
        score = torch.matmul(global_all_feature, lang_feat.unsqueeze(1).transpose(-2, -1))
        score = F.softmax(score, dim=-2)
        global_all_feature = torch.sum(torch.mul(global_all_feature, score), dim=-2) # (64,1024)
        #sum_feat = torch.cat((lang_feat, img_feat, global_all_feature), dim=1)
        #gate = F.sigmoid(self.gate(sum_feat))

        #proj_feat = lang_feat + torch.mul(img_feat, gate) + torch.mul(global_all_feature, 1-gate) # (64,1024)
        proj_feat = lang_feat + img_feat + 0.1 * global_all_feature
        proj_feat = self.proj_norm(proj_feat) # (64,1024)
        proj_feat = torch.sigmoid(self.proj(proj_feat)) # (64,3129)
        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
