'''
-------------------------------------------------------------
Description: Extracts Embeddings based on a configuration file.
    - "save_all_hiddens.yaml", passed as a --config argument in the command line. 
        See documentation in the .yaml file for details
  Automatically copies the configuration file into the output directory of the embeddings.

  Notes:
    - Runs forward passes multiple times to pool in different ways... Takes len(POOL_TYPES) times longer than it should... But, saves on memory! Not an issue if only one type of pooling is used.
-------------------------------------------------------------
'''

import os
import torch
import yaml
import time

import argparse
import datetime

from utils.llm_utils import *


def main(config_filename):
    ## ------------- Set up Parameters from Config File -------------

    with open(config_filename) as stream:
        config = yaml.safe_load(stream)

    ## DIRECTORIES:
    # Static Directories
    MODELS_BASE_DIR = config.get('MODELS_BASE_DIR')
    FINETUNED_BASE_DIR = config.get('FINETUNED_BASE_DIR')
    INPUT_DATA_DIR = config.get('INPUT_DATA_DIR')
    OUTPUT_DATA_DIR = config.get('OUTPUT_DATA_DIR')

    # Models & Checkpoints to use
    BASE_MODEL_PATH = MODELS_BASE_DIR + config.get('BASE_MODEL_NAME')
    USE_BASE_MODEL = config.get('USE_BASE_MODEL', True) 
        # default to get embeddings for base model
    # Only build checkpoint directory if CHECKPOINT_RUN_NAME is not None
    if config.get('CHECKPOINT_RUN_NAME') is not None:
        CHECKPOINT_MODEL_DIR = FINETUNED_BASE_DIR + config.get('CHECKPOINT_RUN_NAME')
    MAX_EPOCHS = config.get('MAX_EPOCHS', None) 
        # default to no adapter paths

    # Run Name, Inputs & Outputs
    RUN_NAME = config.get('RUN_NAME', None) 
    if RUN_NAME is None:
        RUN_NAME = config.get('CHECKPOINT_RUN_NAME')
        assert RUN_NAME is not None, "No run name provided and CHECKPOINT_RUN_NAME is None."
        # default to use same run name as finetuning run, as long as CHECKPOINT_RUN_NAME is provided

    # Date (whether specified or to make current date)
    DATE = config.get('DATE', None)
    if DATE is None:
        DATE = datetime.datetime.now().strftime(format = '%Y-%m-%d')

    DATA_PATH = INPUT_DATA_DIR + config.get('DATA_FILE')
    # Join output directory, run name, and date suffix to create path for saving embeddings output
    SAVE_PATH_ALL = f"{OUTPUT_DATA_DIR}/{RUN_NAME}-{DATE}/{config.get('BASE_MODEL_NAME')}/"


    ## INFERENCE & PROCESSING:
    # Tokenization Params
    BATCH_SIZE = config.get('BATCH_SIZE', 4) 
        # default of 4
    USE_CHAT_TEMPLATE = config.get('USE_CHAT_TEMPLATE', True) 
        # default to use chat template
    PREPROMPT = config.get("PREPROMPT", '')
        # Default to no preprompt

    # Data Loading Params
    NUM_OBS = config.get('NUM_OBS', None) 
        # default to full dataset
    INPUT_COLNAME = config.get('INPUT_COLNAME', None) 
        # default to 'input'
    OUTPUT_COLNAME = config.get('OUTPUT_COLNAME', None) 
        # default to 'output', UNUSED!!
    ID_COLNAME = config.get('ID_COLNAME', None) 
        # default to None, builds own ID's based on index in batch (TBD if this works nicely with parallel dataloader...)

    # Embedding & Pooling Params
    RETURN_MASK = config.get('RETURN_MASK', False) # default to no mask
    POOL_TYPES = config.get('POOL_TYPES')
    LAST_LAYER_ONLY = config.get('LAST_LAYER_ONLY', True) # default to only last layer

    ## ------------- Find Valid Adapter Paths -------------

    if USE_BASE_MODEL:
        ADAPTER_PATHS = [None]
        SAVE_PATHS = [f"{SAVE_PATH_ALL}/base_model/"]
    else:
        ADAPTER_PATHS = []
        SAVE_PATHS = []

    if MAX_EPOCHS is not None:
        assert config.get('CHECKPOINT_RUN_NAME') is not None, "MAX_EPOCHS is not None but no CHECKPOINT_RUN_NAME is not provided"
        for i in range(MAX_EPOCHS):
            # Check that model checkpoint exists befire adding to list of checkpoints
            if os.path.exists(f'{CHECKPOINT_MODEL_DIR}/epoch_{i}'):
                # Add adapter and save path to those to extract embeddings for
                ADAPTER_PATHS.append(f'{CHECKPOINT_MODEL_DIR}/epoch_{i}')
                SAVE_PATHS.append(f'{SAVE_PATH_ALL}/epoch_{i}')


    # Check that output directories exist, else make them
    for SAVE_PATH in SAVE_PATHS:
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)


    # Copy config file into output directory of embedding extraction run
    os.system(f"cp {config_filename} {SAVE_PATH_ALL}/")


    # Print Parameters
    print("------------ Configuration Parameters ------------")
    for key in config.keys():
        print(f"{key}: {config[key]}")



    # Print Adapter Paths and Save Paths
    print("------------ Models and Checkpoints to be Used ------------")
    for i in range(len(ADAPTER_PATHS)):
        print(f"Adapter Path: {ADAPTER_PATHS[i]} \n     will be saved to: {SAVE_PATHS[i]}")



    print("------------ Starting: Extract Hidden States ------------")
    full_start_time = time.time()

    # Load prompts and targets
    prompts, targets, prompt_ids = load_prompts_and_targets(DATA_PATH, num_obs=NUM_OBS, 
                                                            input_colname=INPUT_COLNAME, output_colname=OUTPUT_COLNAME,
                                                            id_colname=ID_COLNAME)

    if prompt_ids is None:
        prompt_ids = [f"obs_{i}" for i in range(len(prompts))]  # Generate IDs for each observation

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loop over candidate model checkpoints (each corresponding to a different adapter)
    # We do this first to avoid loading the model multiple times, which is expensive.
    for i in range(len(ADAPTER_PATHS)):

        SAVE_PATH = SAVE_PATHS[i]
        ADAPTER_PATH = ADAPTER_PATHS[i]

        print(f"\n------------\nUsing model with adapter {ADAPTER_PATH}")
        #print(f"Saving to {SAVE_PATH}")

        # Directories
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        # Load model and tokenizer
        tokenizer, model = load_model(BASE_MODEL_PATH, adapter_path=ADAPTER_PATH)

        # Loop over pooling types specified in POOL_TYPES
        for POOL_TYPE in POOL_TYPES:
            print(f"\n    Pooling type: {POOL_TYPE}")
            start_time = time.time()

            # Get embeddings
            embeds, mask = get_embeddings(model, tokenizer, prompts, 
                                        preprompt=PREPROMPT,
                                        use_chat_template=USE_CHAT_TEMPLATE, pool=POOL_TYPE,
                                        batch_size=BATCH_SIZE, return_mask=RETURN_MASK,
                                        last_layer_only=LAST_LAYER_ONLY)

            # Move to CPU
            # TODO - Make this optional in the future
            embeds = embeds.detach().cpu()
            if mask is not None:
                mask = mask.detach().cpu()

            # Save embeddings
            pooled_save_path = f"{SAVE_PATH}/embeds_pooled_{POOL_TYPE}"

            #torch.save(embeds, pooled_save_path+'.pt') # without h5py
            save_tensor_with_ids(pooled_save_path+'.h5', embeds, prompt_ids) # with h5py

            print(f"    Saved pooled embeddings of shape {embeds.shape} to {pooled_save_path}")
            print(f"    Execution time: {round((time.time() - start_time)/60, 2)} mins.")

    print("------------ Extracting Hidden States Complete! ------------")
    print(f"Total Execution time: {round((time.time() - full_start_time)/60, 2)} mins.")


    # Copy Slurm output file to output directory of embedding extraction run
    slurm_jobnum = os.environ['SLURM_JOB_ID']
    os.system(f"cp slurm-{slurm_jobnum}.out {SAVE_PATH_ALL}/")


# TODO - Simplify the below
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type = str, required = True,
                      help = "Configuration file for save_all_hiddens.py")
    
    args = vars(args.parse_args())

    main(args['config'])