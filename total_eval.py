# This script loads either a standalone fine-tuned model or a base model with a LoRA/PEFT adapter.
# Usage examples and output file descriptions are provide in argparse help options.

import argparse
import os
import datetime
import json
import yaml
import math
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    roc_auc_score
)
from sklearn.calibration import calibration_curve

from utils import llm_utils

def main(cfg_file):


        # Read in config file at path 'cfg_file'
    with open(cfg_file) as stream:
        cfg = yaml.safe_load(stream)


    # Get Inputs from cfg file
    BASE_DIR = cfg.get("BASE_DIR")
    BOOKTYPE = cfg.get("BOOKTYPE")
    DATE = cfg.get("DATE")
    TWINGAME_TYPES = cfg.get("TWINGAME_TYPES")
    TWINGAME_PATH = cfg.get("TWINGAME_PATH")
    TWINGAME_DATAFILE = cfg.get("TWINGAME_DATAFILE")
    
    EVAL_DATAFILE = cfg.get("EVAL_DATAFILE")
    
    MODEL_PATH = cfg.get("MODEL_PATH")
    MODEL_NAME = cfg.get("MODEL_NAME")
    ADAPTER_NAMES = cfg.get("ADAPTER_NAMES")

    NUM_OBS = cfg.get("NUM_OBS")
    INPUT_COLNAME = cfg.get("INPUT_COLNAME")
    OUTPUT_COLNAME = cfg.get("OUTPUT_COLNAME")
    ID_COLNAME = cfg.get("ID_COLNAME")

    BATCH_SIZE = cfg.get("BATCH_SIZE")

    USE_CHAT_TEMPLATE = cfg.get("USE_CHAT_TEMPLATE", True)
    NUM_GPUS = cfg.get("NUM_GPUS", 1)
    MAX_LENGTH = cfg.get("MAX_LENGTH", None)
    

    ## LOOP: OVER TWINGAME TYPES
    for TWINGAME_TYPE in TWINGAME_TYPES:
        print(f"Beginning Evaluation for TwinGame: {TWINGAME_TYPE}")

        TWINGAME_DF_PATH = f"{BASE_DIR}/{TWINGAME_PATH}/{TWINGAME_TYPE}/{TWINGAME_DATAFILE}"
        twingame_df = pd.read_csv(TWINGAME_DF_PATH, nrows=NUM_OBS)
        
        eval_dataset_path = f"{BASE_DIR}/{DATE}-{BOOKTYPE}/paired-books/{TWINGAME_TYPE}/{EVAL_DATAFILE}"

        OUTPUT_PATH = f"{BASE_DIR}/{DATE}-{BOOKTYPE}/{TWINGAME_TYPE}-finetune-twingame-{MODEL_NAME}/eval/"
        # Make output directory if it doesn't exist
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        ## LOOP: OVER ADAPTER PATH
        for ADAPTER_NAME in ADAPTER_NAMES:
            print(f"  Using Adapter: {ADAPTER_NAME}")

            # Configure full adapter paths and output filenames
            if ADAPTER_NAME is not None:
                ADAPTER_PATH = f"{BASE_DIR}/{DATE}-{BOOKTYPE}/{TWINGAME_TYPE}-finetune-twingame-{MODEL_NAME}/{ADAPTER_NAME}"
                OUTPUT_NAME = f"{ADAPTER_NAME}"
            else:
                ADAPTER_PATH = None
                OUTPUT_NAME = f"base_model"

            
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and the necessary drivers installed.")

            tokenizer, model = llm_utils.load_model(
                model_path=MODEL_PATH,
                adapter_path=ADAPTER_PATH,
                torch_dtype=torch.bfloat16,
            )

            prompts, targets, _ = llm_utils.load_prompts_and_targets(
                eval_file=eval_dataset_path,
                input_colname=INPUT_COLNAME,
                output_colname=OUTPUT_COLNAME,
                id_colname=ID_COLNAME,
                num_obs=NUM_OBS,
            )

            # Define candidate tokens for binary classification.
            candidates = ["1", "0"]
            # For single-token candidates, take the first token id.
            candidate_token_ids = {
                cand: tokenizer(cand, add_special_tokens=False)["input_ids"][0]
                for cand in candidates
            }

            logits = llm_utils.get_logits(
                model, tokenizer, prompts,
                batch_size=BATCH_SIZE, use_chat_template=USE_CHAT_TEMPLATE
            )

            probs = torch.softmax(logits, dim=1)

            cand1_raw_probs = probs[:, candidate_token_ids["1"]]
            cand0_raw_probs = probs[:, candidate_token_ids["0"]]

            # Delete to free up memory
            del logits
            del probs

            candidate_prob_sum = cand1_raw_probs + cand0_raw_probs

            cand1_norm_probs = cand1_raw_probs / candidate_prob_sum
            cand0_norm_probs = cand0_raw_probs / candidate_prob_sum

            # Predictions need to be a string because targets are strings and sklearn metrics expect string labels
            predictions = ["1" if cand1_norm_probs[i] >= cand0_norm_probs[i] else "0" for i in range(len(cand1_norm_probs))]
            target_probs = [cand1_norm_probs[i].item() if targets[i] == "1" else cand0_norm_probs[i] for i in range(len(targets))]


            # Merge with twingame_df (after copying)
            df_out = twingame_df.copy()
            df_out.loc[:,'raw_p0'] = cand0_raw_probs.tolist()
            df_out.loc[:,'raw_p1'] = cand1_raw_probs.tolist()
            df_out.loc[:,'raw_sum_p0_p1'] = candidate_prob_sum.tolist()
            df_out.loc[:,'norm_p0'] = cand0_norm_probs.tolist()
            df_out.loc[:,'norm_p1'] = cand1_norm_probs.tolist()
            df_out.loc[:,'prediction'] = predictions
            
            # Save to csv
            df_out.to_csv(f"{OUTPUT_PATH}/{OUTPUT_NAME}.csv")
            

            additional_details = []
            for i in range(0, 5):
                detail = {
                    "p(1) (raw)": float(cand1_raw_probs[i]),
                    "p(0) (raw)": float(cand0_raw_probs[i]),
                    "p(1) (norm)": float(cand1_norm_probs[i]),
                    "p(0) (norm)": float(cand0_norm_probs[i]),
                    "p(0) + p(1) (raw)": float(candidate_prob_sum[i]),
                }
                additional_details.append(detail)
        
            # Compute classification metrics.
            accuracy = accuracy_score(targets, predictions)
            precision = precision_score(targets, predictions, pos_label="1", average="binary")
            recall = recall_score(targets, predictions, pos_label="1", average="binary")
            f1 = f1_score(targets, predictions, pos_label="1", average="binary")
            cm = confusion_matrix(targets, predictions, labels=["1", "0"])
            report = classification_report(targets, predictions, labels=["1", "0"])

            with open(f"{OUTPUT_PATH}/{OUTPUT_NAME}.txt", "w") as f:
                f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
                f.write(f"Precision (Positive=1): {precision:.4f}\n")
                f.write(f"Recall (Positive=1): {recall:.4f}\n")
                f.write(f"F1 Score (Positive=1): {f1:.4f}\n")
                f.write("Confusion Matrix:\n")
                f.write(str(cm) + "\n")
                f.write("\nClassification Report:\n")
                f.write(report + "\n\n")

                # Record details for the first 5 examples.
                for i, detail in enumerate(additional_details):
                    f.write(f"Example {i+1}:\n")
                    f.write(f"  p(1) (raw): {detail['p(1) (raw)']:.4f}\n")
                    f.write(f"  p(0) (raw): {detail['p(0) (raw)']:.4f}\n")
                    f.write(f"  p(1) (norm): {detail['p(1) (norm)']:.4f}\n")
                    f.write(f"  p(0) (norm): {detail['p(0) (norm)']:.4f}\n")
                    f.write(f"  p(0) + p(1) (raw): {detail['p(0) + p(1) (raw)']:.4f}\n")

            
if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config", type = str, required = True,
                      help = "Configuration file for make_books.py")
    
    args = vars(args.parse_args())
    
    main(args['config'])

