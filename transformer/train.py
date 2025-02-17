import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from dataset import BilingualDataset
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from tokenizers.trainers import WordLevelTrainer  # create the vocabulary for given sentence
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter

from config import get_config, get_weights_file_path
from tqdm import tqdm

from pathlib import Path

# Function to get all sentences 
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


# Function to build tokenizers 
def get_or_build_tokenizer(config, ds, lang):
    # create tokenizer based on language
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


#Function to get Dataset : opus_book ( name of dataset)
def get_ds(config):
    ds_raw = load_dataset("Helsinki-NLP/opus_books", f"{config['src_lang']}-{config['target_lang']}" ,split='train')

    # build tokenizers
    src_tokenizer = get_or_build_tokenizer(config, ds_raw, config["src_lang"])
    target_tokenizer = get_or_build_tokenizer(config, ds_raw, config["target_lang"])

    #Keep 90% for training and 10% for validation
    train_ds_size = int(0.9*len(ds_raw))
    validation_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, validation_ds_raw = random_split(ds_raw, [train_ds_size, validation_ds_size])


    # create dataset for training and validation
    train_ds = BilingualDataset(train_ds_raw, src_tokenizer, target_tokenizer, config['src_lang'], config['target_lang'], config['seq_len'])
    validation_ds = BilingualDataset(validation_ds_raw, src_tokenizer, target_tokenizer, config['src_lang'], config['target_lang'], config['seq_len'])


    max_len_src = 0
    max_len_target = 0

    for item in ds_raw:
        src_ids = src_tokenizer.encode(item['translation'][config['src_lang']]).ids
        target_ids = target_tokenizer.encode(item['translation'][config['target_lang']]).ids
        max_len_src = max (max_len_src, len(src_ids))
        max_len_target = max (max_len_target, len(target_ids))


    print(f' Max Length of Source Sentence : {max_len_src}')
    print(f' Max Length of Target Sentence : {max_len_target}')


    #Create DataLoaders
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation_ds, batch_size=1, shuffle=True)              #batch_size = 1 ., would like to process each sentence one by one

    return train_dataloader, validation_dataloader, src_tokenizer, target_tokenizer


# Function to build the trnasformer model as per config
"""
config : user configuration
vocab_src_len  : Source Vocabulary size
vocab_target_len : Target  Vocabulary size
"""
def get_model(config, vocab_src_len, vocab_target_len):
    model = build_transformer(vocab_src_len, vocab_target_len, config['seq_len'], config['seq_len'])
    return model



# Function to train the model
def train_model(config):
    #define the device
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Load datasets 
    train_dataloader, validation_dataloader, src_tokenizer, target_tokenizer = get_ds(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size() ).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0 
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model : {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # loss functiom
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1 ).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch{epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)   #(batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)   #(batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)     #(batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)     #(batch, 1, seq_len, seq_len)

            # run the tensor through transformer 
            encoder_output = model.encode(encoder_input, encoder_mask)                                       #( batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask )        #( batch, seq_len, d_model)
            proj_output = model.project(decoder_output)                                                      #(batch , seq_len, target_vocab_size)

            #extract the label from batch
            label = batch['label'].to(device)          # ( batch, seq_len)


            #(batch , seq_len, target_vocab_size)  -->  (batch* seq_len, target_vocab_size)
            loss = loss_fn(proj_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f"loss" : f"{loss.item() : 6.3f}"})

            # log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            #Backpropagation the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # save the model at the end of each epoch : so that it can resume from last run
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save(
            {'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step}, 
            model_filename)
        
if __name__ == '__main__':
    config = get_config()
    train_model(config)








