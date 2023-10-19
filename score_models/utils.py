from numpy import full
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

def similarity(table_embedding, hidden_states, cand_masks):
    all_cand_masks =  cand_masks.unsqueeze(-1).repeat(1,1,table_embedding.shape[-1])
    table_embedding = torch.sum(table_embedding, dim=1)
    hidden_states = torch.sum(hidden_states * all_cand_masks, dim=1) * (1 / torch.sum(cand_masks, -1).unsqueeze(-1))
    similarity = cosine_similarity(table_embedding, hidden_states)

    # from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
    # loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    # similarity = loss(table_embedding, hidden_states)  # By default, use constant weights = 1/number of samples
    return similarity

def RankingLoss(score, summary_score=None, margin=0.01, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)#[1,16]
    zeros = torch.zeros_like(score)#[1,16]
    loss_func = torch.nn.MarginRankingLoss(0.0) # max(0, -y(x1-x2)+margin)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            zeros = torch.zeros_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss

class TableEmbedding(nn.Module):
    def __init__(self, embed_dim, vocab_size, embed_tokens=None):
        super().__init__()
        self.num_head = 4
        self.embed_dim = embed_dim

        if embed_tokens:
            self.word_embeddings = embed_tokens
        else:
            self.word_embeddings = nn.Embedding(vocab_size, self.embed_dim)

        self.mha_rows = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_head)
        self.mha_columns = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_head)
        self.mha_rows_caption = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_head)
        self.mha_columns_caption = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_head)
        self.mha_content_rows = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_head)
        self.mha_content_columns = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_head)
        # self.mha_table = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_head)

        self.norm_rows = nn.LayerNorm(self.embed_dim)
        self.norm_columns = nn.LayerNorm(self.embed_dim)
        self.norm_rows_caption = nn.LayerNorm(self.embed_dim)
        self.norm_columns_caption = nn.LayerNorm(self.embed_dim)
        self.norm_content = nn.LayerNorm(self.embed_dim)
        # self.norm_table = nn.LayerNorm(self.embed_dim)
    
    def forward(self, caption_ids, rows_ids, columns_ids, content_ids, contents_rows_mask, contents_columns_mask, rows_self_mask, columns_self_mask, rows_caption_mask, columns_caption_mask):
        caption_ids = self.word_embeddings(caption_ids)
        rows_ids = self.word_embeddings(rows_ids)
        columns_ids = self.word_embeddings(columns_ids)
        content_ids = self.word_embeddings(content_ids)

        if contents_rows_mask is not None:
            rows_caption_self_mask = torch.cat((rows_caption_mask, rows_self_mask), dim=-1) #[2,162,204], [2,162,42], [2,162,162]
            columns_caption_self_mask = torch.cat((columns_caption_mask, columns_self_mask), dim=-1)#[2,22,64] [2,22,42], [2,22,22]

            rows_caption_pos = (torch.sum(rows_caption_self_mask, dim=-1) != rows_caption_self_mask.shape[-1]).unsqueeze(-1).repeat(1,1,self.embed_dim) #[2,162,1024]
            rows_ids_after_mask = rows_ids * rows_caption_pos # 这里就是把那些行padding的部分给归0，因为这部分后面不会被mask
            rows_caption_self_mask_pos = (torch.sum(rows_caption_self_mask, dim=-1) != rows_caption_self_mask.shape[-1]).unsqueeze(-1).repeat(1,1,rows_caption_self_mask.shape[-1]) #[2,162,204]
            rows_caption_self_mask = rows_caption_self_mask * rows_caption_self_mask_pos #这里就是取消padding部分的mask掩码，防止生成nan这种东西

            columns_caption_pos = (torch.sum(columns_caption_self_mask, dim=-1) != columns_caption_self_mask.shape[-1]).unsqueeze(-1).repeat(1,1,self.embed_dim)
            columns_ids_after_mask = columns_ids * columns_caption_pos
            columns_caption_self_mask_pos = (torch.sum(columns_caption_self_mask, dim=-1) != columns_caption_self_mask.shape[-1]).unsqueeze(-1).repeat(1,1,columns_caption_self_mask.shape[-1])
            columns_caption_self_mask = columns_caption_self_mask * columns_caption_self_mask_pos

            contents_rows_pos = (torch.sum(contents_rows_mask, dim=-1) != contents_rows_mask.shape[-1]).unsqueeze(-1).repeat(1,1,self.embed_dim)
            contents_rows_ids_after_mask = content_ids * contents_rows_pos
            contents_rows_mask_pos = (torch.sum(contents_rows_mask, dim=-1) != contents_rows_mask.shape[-1]).unsqueeze(-1).repeat(1,1,contents_rows_mask.shape[-1]) 
            contents_rows_mask = contents_rows_mask * contents_rows_mask_pos

            contents_columns_pos = (torch.sum(contents_columns_mask, dim=-1) != contents_columns_mask.shape[-1]).unsqueeze(-1).repeat(1,1,self.embed_dim)
            contents_columns_ids_after_mask = content_ids * contents_columns_pos
            contents_columns_mask_pos = (torch.sum(contents_columns_mask, dim=-1) != contents_columns_mask.shape[-1]).unsqueeze(-1).repeat(1,1,contents_columns_mask.shape[-1]) 
            contents_columns_mask = contents_columns_mask * contents_columns_mask_pos

            rows_caption_self_mask = rows_caption_self_mask.repeat(self.num_head, 1, 1)    
            columns_caption_self_mask = columns_caption_self_mask.repeat(self.num_head, 1, 1) 
            contents_rows_mask = contents_rows_mask.repeat(self.num_head, 1, 1)    
            contents_columns_mask = contents_columns_mask.repeat(self.num_head, 1, 1) 
        else:
            rows_caption_self_mask = columns_caption_self_mask = None
            rows_ids_after_mask = rows_ids
            columns_ids_after_mask = columns_ids
            contents_rows_ids_after_mask = content_ids
            contents_columns_ids_after_mask = content_ids


        caption_self_ids = torch.cat((caption_ids, rows_ids), dim=1)
        rows_caption_self_ids = self.mha_rows_caption(rows_ids_after_mask.permute(1,0,2), caption_self_ids.permute(1,0,2), caption_self_ids.permute(1,0,2), attn_mask=rows_caption_self_mask)[0]
        rows_caption_self_ids = rows_caption_self_ids.permute(1,0,2)
        rows_ids = self.norm_rows(rows_ids+rows_caption_self_ids)

        
        caption_self_ids = torch.cat((caption_ids, columns_ids), dim=1)
        columns_caption_self_ids = self.mha_columns_caption(columns_ids_after_mask.permute(1,0,2), caption_self_ids.permute(1,0,2), caption_self_ids.permute(1,0,2), attn_mask=columns_caption_self_mask)[0]
        columns_caption_self_ids = columns_caption_self_ids.permute(1,0,2)
        columns_ids = self.norm_columns(columns_ids+columns_caption_self_ids)
        

        content_rows_ids = self.mha_content_rows(contents_rows_ids_after_mask.permute(1,0,2), rows_ids.permute(1,0,2), rows_ids.permute(1,0,2), attn_mask=contents_rows_mask)[0].permute(1,0,2)
        content_columns_ids = self.mha_content_columns(contents_columns_ids_after_mask.permute(1,0,2), columns_ids.permute(1,0,2), columns_ids.permute(1,0,2), attn_mask=contents_columns_mask)[0].permute(1,0,2)

        content_ids = content_ids + content_columns_ids + content_rows_ids
        content_ids = self.norm_content(content_ids)

        # table_ids = torch.cat((caption_ids, rows_ids, columns_ids, content_ids), dim=1)
        # table_ids = table_ids.permute(1,0,2)
        # table_ids = self.mha_table(table_ids, table_ids, table_ids)[0].permute(1,0,2)
        # table_ids = self.norm_table(table_ids)

        # return table_ids

        return torch.cat((caption_ids, rows_ids, columns_ids, content_ids), dim=1)

        # return content_ids

class TableModel(nn.Module):
    def __init__(self, embed_dim, embed_tokens=None):
        super().__init__()
        self.num_head = 4
        self.embed_dim = embed_dim
        self.mha_cross = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_head)
        self.norm_merge = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states, table_embedding):
        cross_attn = self.mha_cross(hidden_states.permute(1,0,2), table_embedding.permute(1,0,2), table_embedding.permute(1,0,2))[0]
        cross_attn = cross_attn.permute(1,0,2)#[1,1024,768]
        rest_hidden_states = self.norm_merge(hidden_states + cross_attn) #[1,1024,768]
        return rest_hidden_states
