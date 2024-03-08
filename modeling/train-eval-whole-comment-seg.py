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
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import tqdm
from transformers import Pipeline, pipeline

###############################################################################

# Load environment variables
load_dotenv()

# Models used for testing, both fine-tune and semantic logit
BASE_MODELS = {
    "gte": "thenlper/gte-base",
    "bge": "BAAI/bge-base-en-v1.5",
    # "deberta": "microsoft/deberta-v3-base",
    # "bert-multilingual": "google-bert/bert-base-multilingual-cased",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bert-uncased": "google-bert/bert-base-uncased",
    # "distilbert": "distilbert/distilbert-base-uncased",
}
CONTEXT_WINDOW_SIZES = [
    "single",
    "three",
    # "five",
]

# Fine-tune default settings
DEFAULT_HF_DATASET_PATH = "evamxb/whole-comment-seg-dataset"
DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH = Path("autotrain-text-classification-temp/")
DEFAULT_MODEL_MAX_SEQ_LENGTH = 512
FINE_TUNE_COMMAND_DICT = {
    "data_path": DEFAULT_HF_DATASET_PATH,
    "token": os.environ["HF_AUTH_TOKEN"],
    "project_name": str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
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

# Evaluation storage path
EVAL_STORAGE_PATH = Path("model-eval-results/")

# Delete prior results and then remake
shutil.rmtree(EVAL_STORAGE_PATH, ignore_errors=True)
EVAL_STORAGE_PATH.mkdir(exist_ok=True)

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
class EvaluationResults(DataClassJsonMixin):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    time_pred: float


def evaluate(
    model: LogisticRegressionCV | Pipeline,
    x_test: list[np.ndarray] | list[str],
    y_test: list[str],
    model_name: str,
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

    # Print results
    print(
        f"Accuracy: {accuracy}, "
        f"Precision: {precision}, "
        f"Recall: {recall}, "
        f"F1: {f1}, "
        f"Time/Pred: {perf_time}"
    )

    # Create storage dir for model evals
    this_model_eval_storage = EVAL_STORAGE_PATH / model_name
    this_model_eval_storage.mkdir(exist_ok=True)

    # Create confusion matrix display
    cm = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
    )

    # Save confusion matrix
    cm.figure_.savefig(this_model_eval_storage / "confusion.png")

    # Add a "predicted" column
    df["predicted"] = y_pred

    # Find rows of misclassifications
    misclassifications = df[df["label"] != df["predicted"]]

    # If "embedding" is in the columns, drop it
    if "embedding" in misclassifications.columns:
        misclassifications = misclassifications.drop(columns=["embedding"])

    # Save misclassifications
    misclassifications.to_csv(
        this_model_eval_storage / "misclassifications.csv",
        index=False,
    )

    return EvaluationResults(
        model=this_iter_model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        time_pred=perf_time,
    )


###############################################################################

# Store all results
results = []

# Iter over context window sizes for training
for context_window_size in tqdm(
    CONTEXT_WINDOW_SIZES,
    desc="Context window sizes",
):
    # Load the data for that context window
    context_window_examples = pd.read_csv(
        str(
            TRAINING_DATA_DIR
            / f"whole-comment-seg-{context_window_size}-sentence-examples.csv"
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

    # Train each semantic logit model
    for model_short_name, hf_model_path in tqdm(
        BASE_MODELS.items(),
        desc="Semantic logit models",
        leave=False,
    ):
        # Set seed
        np.random.seed(12)
        random.seed(12)

        this_iter_model_name = f"semantic-logit-{model_short_name}-{context_window_size}-sentences"
        print()
        print(f"Working on: {this_iter_model_name}")
        try:
            # Init sentence transformer
            sentence_transformer = SentenceTransformer(hf_model_path)

            # Preprocess all of the text to embeddings
            print("Preprocessing text to embeddings")
            train_df["embedding"] = [
                np.array(embed)
                for embed in sentence_transformer.encode(
                    train_df["text"].tolist(),
                ).tolist()
            ]
            test_df["embedding"] = [
                np.array(embed)
                for embed in sentence_transformer.encode(
                    test_df["text"].tolist(),
                ).tolist()
            ]

            # Train model
            print("Training model")
            clf = LogisticRegressionCV(
                cv=10,
                max_iter=5000,
                random_state=12,
                class_weight="balanced",
            ).fit(
                train_df["embedding"].tolist(),
                train_df["label"].tolist(),
            )

            # Evaluate model
            results.append(
                evaluate(
                    clf,
                    test_df["embedding"].tolist(),
                    test_df["label"].tolist(),
                    this_iter_model_name,
                    test_df,
                ).to_dict(),
            )
            print()

        except Exception as e:
            print(f"Error during: {this_iter_model_name}, Error: {e}")
            results.append(
                {
                    "model": this_iter_model_name,
                    "error_level": "semantic model training",
                    "error": str(e),
                }
            )

    # Train each fine-tuned model
    for model_short_name, hf_model_path in tqdm(
        BASE_MODELS.items(),
        desc="Fine-tune models",
        leave=False,
    ):
        # Set seed
        np.random.seed(12)
        random.seed(12)

        this_iter_model_name = f"fine-tune-{model_short_name}-{context_window_size}-sentences"
        print()
        print(f"Working on: {this_iter_model_name}")
        try:
            # Delete existing temp storage if exists
            if DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH.exists():
                shutil.rmtree(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH)

            # Update the fine-tune command dict
            this_iter_command_dict = FINE_TUNE_COMMAND_DICT.copy()
            this_iter_command_dict["model"] = hf_model_path

            # Train the model
            ft_train(
                this_iter_command_dict,
            )

            # Evaluate the model
            ft_transformer_pipe = pipeline(
                task="text-classification",
                model=str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
                tokenizer=str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
                padding=True,
                truncation=True,
                max_length=DEFAULT_MODEL_MAX_SEQ_LENGTH,
            )
            results.append(
                evaluate(
                    ft_transformer_pipe,
                    test_df["text"].tolist(),
                    test_df["label"].tolist(),
                    this_iter_model_name,
                    test_df,
                ).to_dict(),
            )

        except Exception as e:
            print(f"Error during: {this_iter_model_name}, Error: {e}")
            results.append(
                {
                    "model": this_iter_model_name,
                    "error_level": "fine-tune model training",
                    "error": str(e),
                }
            )

        print()

    # Print results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False).reset_index(
        drop=True
    )
    results_df.to_csv("all-model-results.csv", index=False)
    print("Current standings")
    print(
        tabulate(
            results_df.head(10),
            headers="keys",
            tablefmt="psql",
            showindex=False,
        )
    )

    print()

print()
print("-" * 80)
print()

# Print results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="f1", ascending=False).reset_index(drop=True)
results_df.to_csv("all-model-results.csv", index=False)
print("Final standings")
print(
    tabulate(
        results_df.head(10),
        headers="keys",
        tablefmt="psql",
        showindex=False,
    )
)
