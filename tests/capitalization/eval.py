import argparse
import datetime
import json
import math
import os
import sys
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import llm_utils


def main(cfg_file):
    with open(cfg_file) as stream:
        cfg = yaml.safe_load(stream)

    BASE_DIR = cfg.get("BASE_DIR")
    EVAL_PATH = cfg.get("EVAL_PATH")
    EVAL_FILE = cfg.get("EVAL_FILE")

    eval_dataset_path = f"{BASE_DIR}/{EVAL_PATH}/{EVAL_FILE}"

    OUTPUT_PATH = cfg.get("OUTPUT_PATH")
    OUTPUT_NAME = cfg.get("OUTPUT_NAME")

    # Make output directory if it doesn't exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    MODEL_PATH = BASE_DIR + cfg.get("MODEL_PATH")
    ADAPTER_PATH = None if not cfg.get("ADAPTER_PATH") else (BASE_DIR + cfg.get("ADAPTER_PATH"))
    
    INPUT_COLNAME = cfg.get("INPUT_COLNAME")
    OUTPUT_COLNAME = cfg.get("OUTPUT_COLNAME")
    ID_COLNAME = cfg.get("ID_COLNAME")
    NUM_OBS = cfg.get("NUM_OBS")
    BATCH_SIZE = cfg.get("BATCH_SIZE")
    NUM_ADDITIONAL = cfg.get("NUM_ADDITIONAL")

    USE_CHAT_TEMPLATE = cfg.get("USE_CHAT_TEMPLATE")
    PREPROMPT = cfg.get("PREPROMPT")
    SYSPROMPT = cfg.get("SYSPROMPT")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and the necessary drivers installed")

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

    probs_l = []
    cand1_raw_probs_l = []
    cand0_raw_probs_l = []
    predictions_raw = []
    for prompt, target in zip(prompts, targets):
        proxy_one = prompt.capitalize()
        candidates = [proxy_one, "0"]
        candidate_token_ids = {
            cand: tokenizer(cand, add_special_tokens=False)["input_ids"][0]
            for cand in candidates
        }

        logits = llm_utils.get_logits(
            model, tokenizer, [prompt],
            batch_size=BATCH_SIZE, use_chat_template=USE_CHAT_TEMPLATE,
            preprompt=PREPROMPT, sysprompt=SYSPROMPT
        )

        probs_l.append(torch.softmax(logits, dim=1)[0])

        cand1_raw_probs_l.append(probs_l[-1][candidate_token_ids[proxy_one]])
        cand0_raw_probs_l.append(1 - probs_l[-1][candidate_token_ids[proxy_one]])

        predictions_raw.append(tokenizer.decode([np.argmax(probs_l[-1]).item()]))

    cand1_raw_probs = torch.stack(cand1_raw_probs_l)
    cand0_raw_probs = torch.stack(cand0_raw_probs_l)

    candidate_prob_sum = cand1_raw_probs + cand0_raw_probs

    cand1_norm_probs = cand1_raw_probs / candidate_prob_sum
    cand0_norm_probs = cand0_raw_probs / candidate_prob_sum

    # Predictions need to be a string because targets are strings and sklearn metrics expect string labels
    predictions = ["1" if predictions_raw[i] == tokenizer.decode(tokenizer(targets[i], add_special_tokens=False)["input_ids"][0]) else "0" for i in range(len(cand1_norm_probs))]
    target_probs = [cand1_norm_probs[i].item() for i in range(len(targets))]

    df_out = pd.DataFrame({"index": range(len(cand1_norm_probs))})
    df_out.loc[:,'raw_p0'] = cand0_raw_probs.tolist()
    df_out.loc[:,'raw_p1'] = cand1_raw_probs.tolist()
    df_out.loc[:,'raw_sum_p0_p1'] = candidate_prob_sum.tolist()
    df_out.loc[:,'norm_p0'] = cand0_norm_probs.tolist()
    df_out.loc[:,'norm_p1'] = cand1_norm_probs.tolist()
    df_out.loc[:,'prediction'] = predictions
    df_out.loc[:,'target'] = targets
    
    # Save to csv
    df_out.to_csv(f"{OUTPUT_PATH}/{OUTPUT_NAME}.csv")

    additional_details = []
    for i in range(0, NUM_ADDITIONAL):
        detail = {
            "target": targets[i],
            "prediction": predictions_raw[i],
            "p(1) (raw)": float(cand1_raw_probs[i]),
            "p(0) (raw)": float(cand0_raw_probs[i]),
            "p(1) (norm)": float(cand1_norm_probs[i]),
            "p(0) (norm)": float(cand0_norm_probs[i]),
            "p(0) + p(1) (raw)": float(candidate_prob_sum[i]),
        }
        additional_details.append(detail)

    # Compute classification metrics.
    targets = ["1"]*len(predictions)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, pos_label="1", average="binary")
    recall = recall_score(targets, predictions, pos_label="1", average="binary")
    f1 = f1_score(targets, predictions, pos_label="1", average="binary")
    cm = confusion_matrix(targets, predictions, labels=["1", "0"])
    report = classification_report(targets, predictions, labels=["1", "0"])

    with open(f"{OUTPUT_PATH}/{OUTPUT_NAME}.txt", "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Average p(1) norm: {cand1_norm_probs.mean():.4f}\n")
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
            f.write(f"  Target: {detail['target']}\n")
            f.write(f"  Prediction: {detail['prediction']}\n")
            f.write(f"  p(1) (raw): {detail['p(1) (raw)']:.4f}\n")
            f.write(f"  p(0) (raw): {detail['p(0) (raw)']:.4f}\n")
            f.write(f"  p(1) (norm): {detail['p(1) (norm)']:.4f}\n")
            f.write(f"  p(0) (norm): {detail['p(0) (norm)']:.4f}\n")
            f.write(f"  p(0) + p(1) (raw): {detail['p(0) + p(1) (raw)']:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Configuration file")

    args = parser.parse_args()
    main(args.config)
