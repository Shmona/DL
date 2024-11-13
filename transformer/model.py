import torch
import torch.nn as nn

import math

""" this class is for Input Embedding : 
It takes 2 aruments 
1.  model dimension = d_model : int 
2.  Vocabulary size =  vocab_size : int  
"""
class InputEmbeedings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  #as per paper
    

""" this class is for Positional Embedding : 
It takes 3 aruments 
1.  model dimension = d_model : int 
2.  Sequence length (Maximum length of the Original sentence i.e Token) =  seq_len : int 
3.  Dropout ( to avoid overfit) = dropout : float          
 """

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout : float) -> None:
        super().__init__()
        self.d_model= d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *(-math.log(10000.0)/d_model))

        # apply sine to even pos
        pe[:, 0::2] = torch.sin(pos*div_term)
        # apply cosine to odd pos
        pe[:, 1::2] = torch.cos(pos*div_term)

        # To process in batch : add a new dimension to positional encoding 
        pe = pe.unsqueeze(0)  #  (1, seq_len, d_model)

        # Save positional encoding in the model ( NOT as parameter of model)
        self.register_buffer('pe', pe)

    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], : ]).requires_grad_(False)  # No need to learn while training
        return self.dropout(x)
    


""" this class is for Layer Normalization : 
It takes 1 aruments 
1.  epsilon (to avoid division by zero) = eps : float         
 """

class LayerNormalization(nn.Module):

    def __init__(self, eps : float = 10**-6)->None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))   # Multiplicative term
        self.beta = nn.Parameter(torch.zeros(1))   # Additive term

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        return self.gamma *(x - mean)/(std + self.eps) + self.beta
    

""" this class is for Feed Forward Block : 
It takes 3 aruments 
1.  model dimension = d_model : int 
2.  inner dimension = d_ff : int
3.  Dropout = dropout          
 """

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model : int, d_ff: int , dropout: float )->None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)   # xW1 + b1  = Y
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)   # YW2 + b2
        

    def forward(self, x):
        # ( batch, seq_len, d_model)  ---> ( batch, seq_len, d_ff) 
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
""" this class is for MultiHead Attention Block : 
It takes 3 aruments 
1.  model dimension = d_model : int     ( make sure d_model should be divisible by h)
2.  No. of head  = h : int   
3.  Dropout = dropout          
 """

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model : int, h: int, dropout : float )->None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout

        # make sure d_model is divisible by h
        assert d_model%h == 0, "d_model is not divisible by h"

        self.d_k = d_model//h                   # integer Division
        self.w_q = nn.Linear(d_model, d_model)  #Wq
        self.w_k = nn.Linear(d_model, d_model)  #Wk
        self.w_v = nn.Linear(d_model, d_model)  #Wv

        self.w_o = nn.Linear(d_model, d_model)  #Wo
        self.dropout = nn.Dropout(dropout)

    " staticmethod means we can call SelfAttention method without having an instance of MultiHeadAttentionBlock class"
    @staticmethod
    def SelfAttention(query, key, value, mask, drooput: nn.Dropout):
        d_k = query.shape[-1]

        " @ : MATMUL in pytorch"
        # ( batch, h, seq_len, d_k)   --->   ( batch, h, seq_len, seq_len) 
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)  # ( batch, h, seq_len, seq_len) 

        if drooput is not None:
            attention_scores = drooput(attention_scores)

        return (attention_scores @ value) , attention_scores
    


    def forward(self, q, k, v, mask):
        # ( batch, seq_len, d_model)  ---> ( batch, seq_len, d_ff) 
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # ( batch, seq_len, d_model)  ---> ( batch, seq_len, h, d_k) : split d_model dimension  --->( batch, h, seq_len, d_k) : transpose(1,2)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x , self.attention_scores = MultiHeadAttentionBlock.SelfAttention(query, key, value, mask, self.dropout)


        # ( batch, h, seq_len, d_k)  --->  ( batch, seq_len, h, d_k)  ---> ( batch, seq_len, d_model) 
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        # ( batch, seq_len, d_model) -- > ( batch, seq_len, d_model) 
        return self.w_o(x)

""" this class is for Residual Connection : 
It takes 1 aruments 
1.  Dropout = dropout          
 """


class ResidualConnection(nn.Module):

    def __init__(self, dropout : float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

""" this class is for Encoder Block : 
It takes 1 aruments 
1.  Multihead attention ( also called as self attention as q, k, v are all same as x)= self_attention_block 
2.  Feed Forward ( Add & Norm)=  FeedForwardBlock
3.  Dropout = dropout        
"""

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float)-> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])  # We need 2 residual connections

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](x,  self.feed_forward_block)
        return x 

""" Encoder - Nx times of Encoder Block
"""
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList)-> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)

        return self.norm(x)
    
""" this class is for Decoder Block : 
It takes 3 aruments 
1.  Multihead attention ( also called as self attention as q, k, v are all same as x)= self_attention_block 
2.  Feed Forward ( Add & Norm)=  FeedForwardBlock
3.  Dropout = dropout        
"""

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))


    def forward(self, x, encoder_output, src_mask, target_mask ):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,target_mask))
        x = self.residual_connections[1](x, lambda x : self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x 


""" Decoder - Nx times of Decoder Block
"""
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    
""" this class is for Projection Layer : Needed to project Decoder output (batch, seq_len, d_model) into Vocabulary : 
It takes 2 aruments 
1.  Model Dimension = d_model 
2.  Vocabulary size=  vocab_size       
"""

class ProjectionLayer(nn.Module):

    def __init__(self, d_model:int, vocab_size: int)->None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(self.d_model, self.vocab_size)



    def forward(self, x):
        #(batch, seq_len, d_model) ---> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim= -1)
    


""" Transformer 
"""

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder : Decoder,
                 src_embed: InputEmbeedings, target_embed: InputEmbeedings, 
                 src_pos: PositionalEncoding, target_pos: PositionalEncoding,
                 projection_layer : ProjectionLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer


    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)
    

    def project(self,x):
        return self.projection_layer(x)



# Function to build the transformer 
"""
src_vocab_size : Vocabulary size needed to map input sentence into vectors 
target_vocab_size : Vocabulary size needed to map output sentence into vectors   
src_seq_len  : Sequence length of input sentence 
target_seq_len : Sequence length of output sentence
d_model : Model Dimension ( 512 as per paper)
N : Number of Encoder and Decoder Blocks (6 as per paper) 
H : Number of Heads ( 8 as per paper)
dropout : 0.1 as per paper 
d_ff : Dimension of Feed forward layer ( 2048 as per paper)
"""

def build_transformer(src_vocab_size: int, target_vocab_size: int, 
                      src_seq_len : int, target_seq_len : int, 
                      d_model : int = 512, N:int = 6, H : int = 8,
                      dropout : float = 0.1, d_ff : int = 2048 )-> Transformer:
    
    #create embedding layers
    src_embed = InputEmbeedings(d_model, src_vocab_size)
    target_embed = InputEmbeedings(d_model, target_vocab_size)

    # Create Positional Encoding layers 
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    # Create Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, H, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)


    # Create Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, H, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, H, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)


    # create Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    # create Transformer 
    transformer= Transformer(encoder, decoder, src_embed, target_embed, src_pos, target_pos, projection_layer)

    # Initialize the parameters 
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer



