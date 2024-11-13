import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__ (self, ds,src_tokenizer, target_tokenizer, src_lang , target_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_len = seq_len
        # create token for SOS, EOS, PAD
        self.sos_token = torch.tensor([self.src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([self.src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([self.src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    

    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]                                     # extract original pair from dataset
        src_text = src_target_pair['translation'][self.src_lang]             # extract src text
        target_text = src_target_pair['translation'][self.target_lang]       # extract target text
        
        # convert each text into tokens then each tokens into Tensor ID
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.target_tokenizer.encode(target_text).ids

        #calculate the padding needed 
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2    # -2 : is for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1    # -1 : only for SOS 

        # make sure seq_len covers all sentences present in dataset
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0 :
            raise ValueError('Sentence is too long !')
        

        #Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        #Add SOS to target text
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )


        # output from Decoder  : Add only EOS
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )


        assert encoder_input.size(0)  == self.seq_len
        assert decoder_input.size(0)  == self.seq_len
        assert label.size(0)  == self.seq_len

        return {
            "encoder_input" : encoder_input  ,  # seq_len
            "decoder_input" : decoder_input  ,  # seq_len 
            "encoder_mask"  : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() , # ( 1, 1, seq_len)
            "decoder_mask"  : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # ( 1, 1, seq_len) & (seq_len , seq_len)
            "label" : label , # (seq_len)
            "src_text" : src_text ,
            "target_text" : target_text
        }



def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)  # mask will take upper triangular matrix as one 
    return mask == 0






