"""Fine-tuning d'un modèle de langage sur les récits, à exécuter sur GPU (par exemple Google Colab).

PyTorch et Transformers ne font pas partie des dépendances du projet : ils sont lourds, inutiles
sans GPU, et déjà installés sur Colab. Ils sont importés seulement au moment de l'entraînement.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from complaint_triage.data.prepare import TARGET
from complaint_triage.models.text import TEXT_COLUMN, clean_text


@dataclass(frozen=True)
class TransformerConfig:
    model_name: str = "distilroberta-base"
    # 256 tokens couvrent environ 200 mots : le récit médian (165 mots) est lu en entier,
    # 29 % des récits sont tronqués. Au-delà, le coût de calcul augmente fortement.
    max_length: int = 256
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    eval_every_steps: int = 1000
    seed: int = 0


def _encode(df: pd.DataFrame, tokenizer: Any, max_length: int) -> Any:
    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "text": [clean_text(text) for text in df[TEXT_COLUMN]],
            "label": df[TARGET].astype(int).tolist(),
        }
    )
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def fine_tune_and_score(
    fit: pd.DataFrame,
    stop: pd.DataFrame,
    to_score: pd.DataFrame,
    output_dir: str,
    config: TransformerConfig = TransformerConfig(),  # noqa: B008  objet immuable
) -> pd.DataFrame:
    """Entraîne sur `fit`, garde le meilleur point selon la perte sur `stop`, note `to_score`.

    Si `output_dir` contient déjà des points de sauvegarde, l'entraînement reprend au dernier au
    lieu de repartir de zéro. Sur Colab, placer `output_dir` dans Google Drive permet ainsi de
    survivre à une coupure de session.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    # Transformers 5 ne connaît plus warmup_ratio : le nombre de pas d'échauffement est calculé
    # ici et passé avec warmup_steps, accepté par toutes les versions.
    total_steps = math.ceil(len(fit) / config.batch_size) * config.epochs
    arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_steps=int(config.warmup_ratio * total_steps),
        eval_strategy="steps",
        eval_steps=config.eval_every_steps,
        save_strategy="steps",
        save_steps=config.eval_every_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=200,
        fp16=torch.cuda.is_available(),
        seed=config.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=_encode(fit, tokenizer, config.max_length),
        eval_dataset=_encode(stop, tokenizer, config.max_length),
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint is not None:
        print(f"Reprise à partir de {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    logits = trainer.predict(_encode(to_score, tokenizer, config.max_length)).predictions
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = shifted[:, 1] / shifted.sum(axis=1)
    return pd.DataFrame(
        {"complaint_id": to_score["complaint_id"].to_numpy(), "score": probabilities}
    )
