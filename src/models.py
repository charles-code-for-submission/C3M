
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import *
from utils_data import *


class ConcatenationScorer(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b], dim=1)  
        return self.net(x).squeeze(1) 
    


class StudentModel(nn.Module):
    def __init__(self, ehr_dim=16, hidden_dim=256, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ehr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 16),
        )
        self.classfier = nn.Linear(16, num_classes)

    def forward(self, ehr):
        h = self.fc(ehr)
        return self.classfier(h), h  
    

class TeacherModel(nn.Module):
    def __init__(
        self,
        ehr_dim: int = 16,
        gene_dim: int = 300,
        gene_hidden: int = 16,
        combined_hidden: int = 16,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gene_fc = nn.Sequential(
            nn.Linear(gene_dim, gene_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.combine_fc = nn.Sequential(
            nn.Linear(ehr_dim + gene_hidden, combined_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(combined_hidden, num_classes)

    def forward(self, ehr: torch.Tensor, gene: torch.Tensor):
        h_gene = self.gene_fc(gene)                 
        h       = torch.cat([ehr, h_gene], dim=1)   
        h       = self.combine_fc(h)                 
        logits  = self.classifier(h)                 
        return logits, h
    
class TeacherTransformerModel(nn.Module):
    def __init__(
        self,
        ehr_dim: int = 16,
        gene_dim: int = 300,
        gene_hidden: int = 16,
        combined_hidden: int = 16,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gene_fc = FTTransformerMixAgg()
        self.combine_fc = nn.Sequential(
            nn.Linear(ehr_dim + gene_hidden, combined_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(combined_hidden, num_classes)

    def forward(self, ehr: torch.Tensor, gene: torch.Tensor):
        """
        ehr:  (batch_size, ehr_dim)   # already 16-dim
        gene: (batch_size, gene_dim)  # 300-dim
        """
        h_gene, rec_logits, mask_scores = self.gene_fc(gene)   
        h       = torch.cat([ehr, h_gene], dim=1)    
        h       = self.combine_fc(h)                 
        logits  = self.classifier(h)                 
        return logits, h, rec_logits, mask_scores


class FTTransformerMixAgg(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        output_dim: int = 16,
        num_masks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.projection = nn.Linear(1, d_model)

        self.num_masks   = num_masks
        self.mask_tokens = nn.Parameter(torch.zeros(num_masks, 1, d_model))
        nn.init.xavier_uniform_(self.mask_tokens)

        self.mask_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.reconstruct = nn.Linear(d_model, 2)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor):

        B, N = x.shape
        device = x.device

        gene_emb = self.projection(x.unsqueeze(-1))

        rec_views   = []
        score_views = []
        state_views = []

        for k in range(self.num_masks):
            mask_tok = self.mask_tokens[k].expand(B, -1, -1) 

            tokens = torch.cat([mask_tok, gene_emb], dim=1)  
            _, attn_w = self.mask_attn(
                query=tokens[:, :1, :],  
                key=tokens[:, 1:, :],   
                value=tokens[:, 1:, :]
            ) 
            mask_scores = attn_w.squeeze(1)

            m = mask_scores.unsqueeze(-1)      
            mask_tok_exp = mask_tok.expand(-1, N, -1)
            masked_genes = (1 - m) * gene_emb + m * mask_tok_exp

            seq = torch.cat([mask_tok, masked_genes], dim=1)
            H   = self.transformer(seq)                       

            rec_views.append(self.reconstruct(H[:, 1:, :])) 
            state_views.append(H[:, 0, :])                
            score_views.append(mask_scores)           

        rec_stack    = torch.stack(rec_views,   dim=0)  
        rec_logits   = rec_stack.mean(dim=0)           

        score_stack  = torch.stack(score_views, dim=0) 
        mask_scores  = score_stack.mean(dim=0)        

        state_stack  = torch.stack(state_views, dim=0) 
        mask_states  = state_stack.mean(dim=0)         

        out = self.head(self.norm(mask_states))       
        return out, rec_logits, mask_scores



class FTTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        output_dim: int = 16,
        dropout: float = 0.1    # 0.1 best
    ):
        super().__init__()
        self.projection = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_features]
        returns: [B, output_dim]
        """
        B, N = x.shape 
        
        x = x.unsqueeze(-1)                   
        tokens = self.projection(x)            

        tokens = tokens.transpose(0, 1)        
        encoded = self.transformer(tokens)     
        encoded = encoded.transpose(0, 1)      
        pooled = encoded.mean(dim=1)
        out = self.head(self.norm(pooled))
        return out

class TeachformerModel(nn.Module):
    def __init__(
        self,
        ehr_dim: int = 16,
        gene_dim: int = 300,
        gene_hidden: int = 16,
        combined_hidden: int = 16,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gene_fc = FTTransformer()
        self.combine_fc = nn.Sequential(
            nn.Linear(ehr_dim + gene_hidden, combined_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(combined_hidden, num_classes)

    def forward(self, ehr: torch.Tensor, gene: torch.Tensor):
        """
        ehr:  (batch_size, ehr_dim)   # already 16-dim
        gene: (batch_size, gene_dim)  # 300-dim
        """
        h_gene = self.gene_fc(gene)                  
        h       = torch.cat([ehr, h_gene], dim=1)    
        h       = self.combine_fc(h)                 
        logits  = self.classifier(h)                 
        return logits, h
    
class AttentionLayer2Coarse(nn.Module):
    def __init__(self, in_features, in_features2, out_features, use_mlp=False, dropout=0.3):
        super(AttentionLayer2Coarse, self).__init__()
        
        self.query = nn.Linear(in_features, out_features)
        self.key = nn.Linear(in_features2, out_features)
        self.value = nn.Linear(in_features2, out_features)
        self.K = 20
        self.use_mlp = use_mlp
#         self.aft_mlp = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(p=dropout)
#         self.act = nn.ELU()
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.temp = 0.5
        use_mlp_attention = True
        self.use_mlp_attention = use_mlp_attention
        if self.use_mlp_attention:
            self.mlp_attention = nn.Sequential(
                nn.Linear(2 * out_features, out_features),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(out_features, 1)
            )
            
    def forward(self, patient_embedding, gene_embedding):
        B = patient_embedding.size(0)
        Q = self.query(patient_embedding)                 
        K_mat = self.key(gene_embedding)                 
        V_mat = self.value(gene_embedding)               
        D = Q.size(1)
        N = K_mat.size(0)

        Q_exp = Q.unsqueeze(1).expand(-1, N, -1)          
        K_exp = K_mat.unsqueeze(0).expand(B, -1, -1)      
        qk_cat = torch.cat([Q_exp, K_exp], dim=-1)       
        sim = self.mlp_attention(qk_cat).squeeze(-1)     

        logits = sim / self.temp                         
        K_sel = self.K

        logits_k = logits.unsqueeze(1).expand(-1, K_sel, -1)  
        logits_flat = logits_k.reshape(-1, N)                

        masks_flat = F.gumbel_softmax(logits_flat, tau=self.temp, hard=True, dim=-1)  

        masks = masks_flat.view(B, K_sel, N)                  

        value_exp = V_mat.unsqueeze(0).unsqueeze(0).expand(B, K_sel, N, D)
        masks_exp = masks.unsqueeze(-1)
        topk_values = (masks_exp * value_exp).sum(dim=2)     

        Q1 = Q.unsqueeze(1)                                  
        scores = (Q1 * topk_values).sum(-1) / math.sqrt(D)     
        weights = F.softmax(scores, dim=-1)                    
        weights = self.dropout(weights)
        context = (weights.unsqueeze(-1) * topk_values).sum(dim=1)  

        return context


    
class Predictor(nn.Module):  
    def __init__(self, hidden_dims, dropout=0.2):
        super(Predictor, self).__init__()
        layers = []
        
        if len(hidden_dims) >=1:
            prev_dim = hidden_dims[0]
            for dim in hidden_dims[1:]:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.LeakyReLU(negative_slope=0.1))
                layers.append(nn.Dropout(dropout))

                prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 2))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class EHREncoder(nn.Module): 
    def __init__(self, gdataset, in_channels, hidden_channels, out_channels, dropout,\
                eid_emb,  ehr_dim,\
                 ehr_hidden_dims):    
        super(EHREncoder, self).__init__()

        gdata = gdataset.data

        num_nodes = gdata.x.shape[0]

        self.ehr_emb = EHREmb(ehr_dim, ehr_hidden_dims, activation='leaky_relu', dropout=dropout) 
        self.ehr_emb2 = EHRGNN_motor(init_emb=eid_emb, num_nodes=num_nodes, in_channels=in_channels, hidden_channels=hidden_channels,\
                        out_channels=out_channels, dropout=dropout)  
        

class EHREmb(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation='relu', dropout=0.2):
        super(EHREmb, self).__init__()
        assert len(hidden_dims) >= 1, "At least one hidden layer is required."
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Dropout(dropout))

            if activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class EHRGNN_motor(torch.nn.Module):
    def __init__(self, init_emb, num_nodes, in_channels, hidden_channels, out_channels, dropout):
        super(EHRGNN_motor, self).__init__()
        
        eid_num = init_emb.shape[0]
        eid_dim = init_emb.shape[-1]
        # self.init_emb = torch.tensor(init_emb)
        self.init_emb = init_emb
        self.map = nn.Linear(eid_dim, in_channels)

        self.s_embedding = nn.Embedding(num_nodes - eid_num , in_channels)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.convs = [self.conv1, self.conv2]
        self.dropout = dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.s_embedding.reset_parameters()


    def forward(self, adj_t):

        eid_x = self.map(self.init_emb)

        snp_x = self.s_embedding(torch.arange(0, self.s_embedding.weight.shape[0], device=adj_t.device))
        x = torch.cat([eid_x, snp_x], axis=0)
        for l, conv in enumerate(self.convs[:-1]):

            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    

class LinkPredictionModel(nn.Module):
    def __init__(self, gnn_input_dim, hidden_dim, dropout):
        super(LinkPredictionModel, self).__init__()
        self.fc1 = nn.Linear(gnn_input_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, 1)  
        self.dropout=dropout
    def forward(self, node1_embed, node2_embed):
        
        x = torch.cat([node1_embed, node2_embed], dim=-1)
        x = self.fc1(x) 
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=self.dropout)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
class AttentionLayer(nn.Module):
    def __init__(self, q_inputdim, v_inputdim, out_features, use_mlp=False, dropout=0.3):
        super(AttentionLayer, self).__init__()
        
        self.query = nn.Linear(q_inputdim, out_features)
        self.key = nn.Linear(v_inputdim, out_features)
        self.value = nn.Linear(v_inputdim, out_features)
        
        self.use_mlp = use_mlp
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, patient_embedding, gene_embedding):
        if self.use_mlp is True:
            query = self.query(patient_embedding)
            key = self.key(gene_embedding)
            value = self.value(gene_embedding)
        else:
            query = patient_embedding
            key = gene_embedding
            value = gene_embedding
        
        attention_weights = torch.matmul(query, key.T) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_weights, dim=1) 

        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, value)
        attention_output = self.act(attention_output)
        return attention_output
