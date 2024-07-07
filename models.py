import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import normalize


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 750):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class HistoryUnit(torch.nn.Module):
    def __init__(self, opt):
        super(HistoryUnit, self).__init__()
        self.n_feature=opt["feat_dim"] 
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_hist_dec_head = 4
        n_hist_dec_layer = 5
        n_hist_dec_head_2 = 4
        n_hist_dec_layer_2 = 2
        self.anchors=opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        

        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)   

        self.history_encoder_block1 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_hist_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_hist_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  
        
        
        self.history_encoder_block2 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_hist_dec_head_2, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_hist_dec_layer_2, 
                                            nn.LayerNorm(n_embedding_dim))  
        
        

        self.snip_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim//4), nn.ReLU())     
        self.snip_classifier = nn.Sequential(nn.Linear(self.history_tokens*n_embedding_dim//4, (self.history_tokens*n_embedding_dim//4)//4), nn.ReLU(), nn.Linear((self.history_tokens*n_embedding_dim//4)//4,n_class))                      
        

        self.history_token = nn.Parameter(torch.zeros(self.history_tokens, 1, n_embedding_dim))
        # self.history_token_extra = nn.Parameter(torch.zeros(self.history_tokens*2, 1, n_embedding_dim))

        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout2 = nn.Dropout(0.1)


    def forward(self, long_x, encoded_x):
        

        ## History Encoder
        hist_pe_x = self.history_positional_encoding(long_x)
        history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)  
        hist_encoded_x_1 = self.history_encoder_block1(history_token, hist_pe_x)
        hist_encoded_x_2 = self.history_encoder_block2(hist_encoded_x_1, encoded_x)
        hist_encoded_x_2 = hist_encoded_x_2 + self.dropout2(hist_encoded_x_1)
        hist_encoded_x = self.norm2(hist_encoded_x_2)
   
        ## Snippet Classfication Head
        snippet_feat = self.snip_head(hist_encoded_x_1)
        snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
        
        snip_cls = self.snip_classifier(snippet_feat)
        
        return hist_encoded_x, snip_cls



class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature=opt["feat_dim"] 
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_enc_layer=opt["enc_layer"]
        n_enc_head=opt["enc_head"]
        n_dec_layer=opt["dec_layer"]
        n_dec_head=opt["dec_head"]
        n_comb_dec_head = 4
        n_comb_dec_layer = 5
        n_seglen=opt["segment_size"]
        self.anchors=opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0

        self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
        self.encoder = nn.TransformerEncoder(
                                            nn.TransformerEncoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_enc_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_enc_layer, 
                                            nn.LayerNorm(n_embedding_dim))
                                            
        self.decoder = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  

        self.history_unit = HistoryUnit(opt)


        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_comb_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_comb_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  
            

        self.classifier = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,n_class))
        self.regressor = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,2))    
                           
        
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))


        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb,base_x_flow],dim=-1)
        
        base_x = base_x.permute([1,0,2])# seq_len x batch x featsize x 

        short_x = base_x[-self.short_window_size:]

        long_x = base_x[:-self.short_window_size]
        
        ## Anchor Feature Generator
        pe_x = self.positional_encoding(short_x)
        encoded_x = self.encoder(pe_x)   
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
        decoded_x = self.decoder(decoder_token, encoded_x) 
        decoded_x = decoded_x

        ## Future-Supervised History Module
        hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)


        ## History Driven Anchor Refinement
        decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
        decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
        decoded_anchor_feat = self.norm1(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
        # Predition Module
        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)
        
        return anc_cls, anc_reg, snip_cls

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
        
