from pathlib import Path

def get_config():
    return{
        "batch_size": 8,
        "num_epochs": 20,
        "lr":10**-4,
        "seq_len": 350,
        "src_lang": "en",
        "target_lang": "it",
        "model_folder": "weights",
        "model_basename": "transformer_model_",
        "preload":None, 
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/transformer_model_"

    }

def get_weights_file_path(config, epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.')/model_folder/model_filename)
