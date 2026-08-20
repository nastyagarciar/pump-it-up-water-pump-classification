"""Model helpers for the Pump It Up portfolio project."""

from __future__ import annotations

from catboost import CatBoostClassifier


def build_catboost_model(
    *,
    iterations: int = 1000,
    learning_rate: float = 0.1,
    depth: int = 6,
    l2_leaf_reg: float = 5.0,
    random_seed: int = 42,
) -> CatBoostClassifier:
    """Create the CatBoost configuration used for validation experiments."""
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        verbose=False,
        thread_count=-1,
    )


def build_final_catboost_model() -> CatBoostClassifier:
    """Create the final full-training CatBoost configuration."""
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        iterations=800,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )
