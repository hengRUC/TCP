import random
import torch
from torch import nn
import torch.nn.functional as F
from models.bert import (
    BertConfig, BertModel, BertOnlyMLMHead, BertOnlyNSPHead, BertForMaskedLM)
from models.video_encoder import SwinTransformer3D
from models.text_encoder import TextEncoderForPretraining

from tools.soft_dtw import SoftDTW
from tools.Brownian_bridge import PRT


class VideoTokenPos(nn.Module):
    def __init__(self,num_patches=6, num_frames=32, hidden_size=768):
        super().__init__()
        self.s_pos_embed = nn.Parameter(0.02*torch.randn(1, 1, num_patches, hidden_size), requires_grad=True)
        self.t_pos_embed = nn.Parameter(0.02*torch.randn(1, num_frames, 1, hidden_size), requires_grad=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, video_embd):
        video_embd = video_embd + self.s_pos_embed + self.t_pos_embed
        video_embd = self.norm(video_embd)
        return video_embd

class SentEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.hidden_size
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.segment_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(cfg.max_position_embeddings).expand((1, -1)))

    def forward(self, inputs_embeds, token_type_ids):
        segment_embeddings = self.segment_embeddings(token_type_ids) # B, N, C
        seq_length = inputs_embeds.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings =  inputs_embeds + position_embeddings + segment_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TCP_Pretrain(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.cfg = config
        self.args = args
        self.video_encoder = SwinTransformer3D(**config.VideoEncoder)
        bert_config = BertConfig.from_json_file(config.bert_config)

        self.text_encoder = TextEncoderForPretraining(args, config=bert_config)

        self.video_downsample = nn.MaxPool2d((2,3), stride=(1,1))

        self.video_local_proj = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.text_local_proj = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)

        self.video_global_proj = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.text_global_proj = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)

        setattr(bert_config,'type_vocab_size',config.type_vocab_size)
        self.sent_embedding = SentEmbedding(bert_config)

    def _init_sent_embedding(self):
        self.sent_embedding.position_embeddings.weight.data.copy_(self.text_encoder.bert.embeddings.position_embeddings.weight.data)

    # Part I: Cross-modal Sequence Alignment with Soft-DTW Cost
    def Soft_DTW(self, video_seq, text_seq):
        vb, vm, clip, vc = video_seq.shape
        tb, tm, tc = text_seq.shape

        selected_clip = []  
        for i in range(vb):
            random_clip_index = random.randint(0, clip - 1)
            random_clip = selected_clip[i, random_clip_index, :]
            selected_clip.append(random_clip)
        video_seq = torch.tensor(selected_clip)

        sdtw = SoftDTW(use_cuda=True, gamma=self.cfg.TRAINING.lambda)
        loss = sdtw(video_seq, text_seq)
        return loss

    # Part II: Intra-modal Sequence Modeling  with Process-wised Regularization
    def PRT_term(self, video_seq, video_inx, text_seq):
        vb, vm, clips, vc = video_seq.shape
        tb, tm, tc = text_seq.shape
        assert vm==tm

        cur_prt = PRT(self.arg)
        text_inx = [i for i in range(tm)]
        t_loss = cur_prt(text_seq, text_inx)

        for i in range(clips):
            t_loss += cur_prt(video_seq[:,i,:], video_inx[:,i])

        return t_loss


    def downsample_video_embd(self, video_embd):
        sample_clip = self.cfg.DATA.sample_clip
        B, N, H, W, C = video_embd.size() # B, N, H, W, C
        video_embd = video_embd.permute(0,1,4,2,3)
        video_embd = self.video_downsample(video_embd.view(B*N, C, H, W))
        video_embd = video_embd.permute(0,2,3,1) # B*N, H, W, C
        video_embd = video_embd.view(B, N, video_embd.size(-3), video_embd.size(-2),video_embd.size(-1))
        video_embd = video_embd.flatten(2,3) # B, N, X, C

        video_feat = video_embd.view(B, sample_clip, int(N/sample_clip), -1, C)
        video_feat = video_feat.mean(dim=[2,3])

        return video_feat, video_embd


    def forward(self, video_frames, text_ids, 
                    attention_mask,  mlm_labels = None, 
                    stage=2,is_train=True,is_pretrain_val=False):

        # extract video feature
        B, C, N, H, W = video_frames.size()
        video_global_embd, video_local_embd = self.video_encoder(video_frames) # B, N, H, W, C

        video_local_feat1, _ = self.downsample_video_embd(video_local_embd)

        # extract text feature
        B,M,L = text_ids.shape
        text_local_embd = self.text_encoder(text_ids.view(B*M, L), attention_mask=attention_mask.view(B*M, L), return_dict=True, stage=0).view(B, M, L, -1) # B, M, L, C

        text_local_feat = text_local_embd[:,:,0,:] # B, M, C
        video_local_feat = F.normalize(self.video_local_proj(video_local_feat1),dim=-1)
        text_local_feat = F.normalize(self.text_local_proj(text_local_feat),dim=-1)

        B,M,L,C = text_local_embd.shape
        text_segment_id = torch.arange(M, device=text_local_embd.device).repeat(B,1).repeat_interleave(L,dim=1)# B, N
        text_local_embd = self.sent_embedding(text_local_embd.view(B,M*L,-1), text_segment_id)


        soft_dtw_loss, prt_vt_loss = 0, 0
        soft_dtw_loss = self.Soft_DTW(video_local_feat, text_local_feat)       
        prt_vt_loss = self.PRT_term(video_local_feat,self.video_inx,text_local_feat)
        weight=self.cfg.TRAINING.PRT_weight
 
        return soft_dtw_loss + weight*prt_vt_loss
