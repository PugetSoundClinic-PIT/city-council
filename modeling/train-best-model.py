import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from autotrain.trainers.text_classification.__main__ import train as ft_train
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import Pipeline, pipeline

###############################################################################

# Load environment variables
load_dotenv()

BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Fine-tune default settings
DEFAULT_HF_DATASET_PATH = "evamxb/whole-comment-seg-dataset"
FT_MODEL_NAME = "whole-pc-section-window-classifier"
FT_MODEL_STORAGE_PATH = Path(f"{FT_MODEL_NAME}/")
DEFAULT_MODEL_MAX_SEQ_LENGTH = 512
FT_PARAMS = {
    "data_path": DEFAULT_HF_DATASET_PATH,
    "token": os.environ["HF_AUTH_TOKEN"],
    "project_name": str(FT_MODEL_STORAGE_PATH),
    "model": BASE_MODEL,
    "text_column": "text",
    "target_column": "label",
    "train_split": "train",
    "valid_split": "valid",
    "epochs": 3,
    "lr": 5e-5,
    "auto_find_batch_size": True,
    "seed": 12,
    "max_seq_length": DEFAULT_MODEL_MAX_SEQ_LENGTH,
}

TRAINING_DATA_DIR = Path(__file__).parent / "training-data"

# Set seed
np.random.seed(12)
random.seed(12)

###############################################################################


def cast_split_to_dataset(
    df: pd.DataFrame,
    num_classes: int,
    class_labels: list[str],
) -> tuple[datasets.Dataset]:
    # Construct features for the dataset
    features = datasets.Features(
        text=datasets.Value("string"),
        label=datasets.ClassLabel(
            num_classes=num_classes,
            names=class_labels,
        ),
    )

    # Construct the dataset
    return datasets.Dataset.from_pandas(
        df,
        features=features,
        preserve_index=False,
    )


@dataclass
class EvaluationResults:
    accuracy: float
    precision: float
    recall: float
    f1: float
    time_pred: float


def evaluate(
    model: Pipeline,
    x_test: list[np.ndarray] | list[str],
    y_test: list[str],
    df: pd.DataFrame,
) -> EvaluationResults:
    # Evaluate the model
    print("Evaluating model")

    # Recore perf time
    start_time = time.time()
    y_pred = model.predict(x_test)
    perf_time = (time.time() - start_time) / len(y_test)

    # Get the actual predictions from Pipeline
    if isinstance(model, Pipeline):
        y_pred = [pred["label"] for pred in y_pred]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
    )

    # Create confusion matrix display
    cm = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
    )

    # Save confusion matrix
    cm.figure_.savefig(f"{FT_MODEL_NAME}-confusion.png")

    # Add a "predicted" column
    df["predicted"] = y_pred

    # Find rows of misclassifications
    misclassifications = df[df["label"] != df["predicted"]]

    # If "embedding" is in the columns, drop it
    if "embedding" in misclassifications.columns:
        misclassifications = misclassifications.drop(columns=["embedding"])

    # Save misclassifications
    misclassifications.to_csv(
        f"{FT_MODEL_NAME}-misclassifications.csv",
        index=False,
    )

    return EvaluationResults(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        time_pred=perf_time,
    )


###############################################################################

# Load the data for that context window
context_window_examples = pd.read_csv(
    str(
        TRAINING_DATA_DIR
        / f"whole-comment-seg-three-sentence-examples.csv"
    ),
)

# Drop session_id column
context_window_examples = context_window_examples.drop(columns=["session_id"])

# Create splits
train_df, test_and_valid_df = train_test_split(
    context_window_examples,
    test_size=0.4,
    shuffle=True,
    stratify=context_window_examples["label"],
)
valid_df, test_df = train_test_split(
    test_and_valid_df,
    test_size=0.5,
    shuffle=True,
    stratify=test_and_valid_df["label"],
)

# Create a dataframe where the rows are the different splits and there are three columns
# one column is the split name, the other columns are the counts of match
split_counts = []
for split_name, split_df in [
    ("train", train_df),
    ("valid", valid_df),
    ("test", test_df),
]:
    split_counts.append(
        {
            "split": split_name,
            **split_df["label"].value_counts().to_dict(),
        }
    )
split_counts_df = pd.DataFrame(split_counts)
print("Split counts:")
print(split_counts_df)
print()

# Store class details required for feature construction
num_classes = context_window_examples["label"].nunique()
class_labels = list(context_window_examples["label"].unique())

# Create the datasets
train_ds = cast_split_to_dataset(
    train_df,
    num_classes,
    class_labels,
)
valid_ds = cast_split_to_dataset(
    valid_df,
    num_classes,
    class_labels,
)
test_ds = cast_split_to_dataset(
    test_df,
    num_classes,
    class_labels,
)

# Store as dataset dict
ds_dict = datasets.DatasetDict(
    {
        "train": train_ds,
        "valid": valid_ds,
        "test": test_ds,
    }
)

# Push to hub
print("Pushing this fieldset to hub")
ds_dict.push_to_hub(
    DEFAULT_HF_DATASET_PATH,
    private=True,
    token=os.environ["HF_AUTH_TOKEN"],
)
print()
print()

# Set seed
np.random.seed(12)
random.seed(12)

# Delete existing temp storage if exists
if FT_MODEL_STORAGE_PATH.exists():
    shutil.rmtree(FT_MODEL_STORAGE_PATH)

# Train the model
ft_train(FT_PARAMS)

# Evaluate the model
ft_transformer_pipe = pipeline(
    task="text-classification",
    model=str(FT_MODEL_STORAGE_PATH),
    tokenizer=str(FT_MODEL_STORAGE_PATH),
    padding=True,
    truncation=True,
    max_length=DEFAULT_MODEL_MAX_SEQ_LENGTH,
)

eval_results = evaluate(
    ft_transformer_pipe,
    test_df["text"].tolist(),
    test_df["label"].tolist(),
    test_df,
)

print(eval_results)