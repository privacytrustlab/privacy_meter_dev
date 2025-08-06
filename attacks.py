from typing import Any

import numpy as np
from sklearn.metrics import auc, roc_curve


def get_rmia_out_signals(
    all_signals: np.ndarray,
    all_memberships: np.ndarray,
    target_model_idx: int,
    num_reference_models: int,
) -> np.ndarray:
    """
    Get average prediction probability of samples over offline reference models (excluding the target model).

    Args:
        all_signals (np.ndarray): Softmax value of all samples in every model.
        all_memberships (np.ndarray): Membership matrix for all models (if a sample is used for training a model).
        target_model_idx (int): Target model index.
        num_reference_models (int): Number of reference models used for the attack.

    Returns:
        np.ndarray: Average softmax value for each sample over OUT reference models.
    """
    paired_model_idx = (
        target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
    )
    # Add non-target and non-paired model indices
    columns = [
        i
        for i in range(all_signals.shape[1])
        if i != target_model_idx and i != paired_model_idx
    ][: 2 * num_reference_models]
    selected_signals = all_signals[:, columns]
    non_members = ~all_memberships[:, columns]
    out_signals = selected_signals * non_members
    # Sort the signals such that only the non-zero signals (out signals) are kept
    out_signals = -np.sort(-out_signals, axis=1)[:, :num_reference_models]
    return out_signals


def tune_offline_a(
    target_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    logger: Any,
    attack_function=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Fine-tune coefficient offline_a used in RMIA or other attacks.

    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in two models (target and reference).
        population_signals (np.ndarray): Population signals.
        all_memberships (np.ndarray): Membership matrix for all models.
        logger (Any): Logger object for the current run.
        attack_function (callable, optional): Attack function to use for tuning. Defaults to run_rmia.

    Returns:
        float: Optimized offline_a obtained by attacking a paired model with the help of the reference models.
    """
    if attack_function is None:
        attack_function = run_rmia
        
    paired_model_idx = (
        target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
    )
    logger.info(f"Fine-tuning offline_a using paired model {paired_model_idx}")
    paired_memberships = all_memberships[:, paired_model_idx]
    offline_a = 0.0
    max_auc = 0
    mia_scores_array = None
    membership_array = None
    last_mia_scores = None
    
    for test_a in np.arange(0, 1.1, 0.1):
        mia_scores = attack_function(
            paired_model_idx,
            all_signals,
            population_signals,
            all_memberships,
            1,
            test_a,
        )
        last_mia_scores = mia_scores  # Keep track of the last computed scores
        fpr_list, tpr_list, _ = roc_curve(
            paired_memberships.ravel(), mia_scores.ravel()
        )
        roc_auc = auc(fpr_list, tpr_list)
        if roc_auc > max_auc:
            max_auc = roc_auc
            offline_a = test_a
            mia_scores_array = mia_scores.ravel().copy()
            membership_array = paired_memberships.ravel().copy()
        logger.info(f"offline_a={test_a:.2f}: AUC {roc_auc:.4f}")
    
    # Handle case where no improvement was found
    if mia_scores_array is None and last_mia_scores is not None:
        mia_scores_array = last_mia_scores.ravel().copy()
        membership_array = paired_memberships.ravel().copy()
    elif mia_scores_array is None:
        # Fallback - should not happen in normal execution
        mia_scores_array = np.array([])
        membership_array = np.array([])
    
    # Ensure membership_array is never None
    if membership_array is None:
        membership_array = paired_memberships.ravel().copy()
        
    return offline_a, mia_scores_array, membership_array


def run_rmia(
    target_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    num_reference_models: int,
    offline_a: float,
) -> np.ndarray:
    """
    Attack a target model using the RMIA attack with the help of offline reference models.

    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in the target model.
        population_signals (np.ndarray): Softmax value of all population samples in the target model.
        all_memberships (np.ndarray): Membership matrix for all models.
        num_reference_models (int): Number of reference models used for the attack.
        offline_a (float): Coefficient offline_a is used to approximate p(x) using P_out in the offline setting.

    Returns:
        np.ndarray: MIA score for all samples (a larger score indicates higher chance of being member).
    """
    target_signals = all_signals[:, target_model_idx]
    out_signals = get_rmia_out_signals(
        all_signals, all_memberships, target_model_idx, num_reference_models
    )
    mean_out_x = np.mean(out_signals, axis=1)
    mean_x = (1 + offline_a) / 2 * mean_out_x + (1 - offline_a) / 2
    prob_ratio_x = target_signals.ravel() / mean_x

    z_signals = population_signals[:, target_model_idx]
    population_memberships = np.zeros_like(population_signals).astype(
        bool
    )  # All population data are OUT for all models
    z_out_signals = get_rmia_out_signals(
        population_signals,
        population_memberships,
        target_model_idx,
        num_reference_models,
    )
    mean_out_z = np.mean(z_out_signals, axis=1)
    mean_z = (1 + offline_a) / 2 * mean_out_z + (1 - offline_a) / 2
    prob_ratio_z = z_signals.ravel() / mean_z

    ratios = prob_ratio_x[:, np.newaxis] / prob_ratio_z
    counts = np.average(ratios > 1.0, axis=1)

    return counts


def run_loss(target_signals: np.ndarray) -> np.ndarray:
    """
    Attack a target model using the LOSS attack.

    Args:
        target_signals (np.ndarray): Softmax value of all samples in the target model.

    Returns:
        np.ndarray: MIA score for all samples (a larger score indicates higher chance of being member).
    """
    mia_scores = -target_signals
    return mia_scores


def run_mia_bits(
    target_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    num_reference_models: int,
    offline_a: float,
) -> np.ndarray:
    """
    Attack a target model using the information-theoretic test statistic approach.

    Computes: Test Statistic = log(p(x|θ)/p(x)) + D_KL(p(z) || p(z|θ))

    Args:
        target_model_idx (int): Index of the target model.
        all_signals (np.ndarray): Softmax value of all samples in the target model.
        population_signals (np.ndarray): Softmax value of all population samples in the target model.
        all_memberships (np.ndarray): Membership matrix for all models.
        num_reference_models (int): Number of reference models used for the attack.
        offline_a (float): Coefficient offline_a is used to approximate p(x) using P_out in the offline setting.

    Returns:
        np.ndarray: MIA score for all samples (a larger score indicates higher chance of being member).
    """
    target_signals = all_signals[:, target_model_idx]
    out_signals = get_rmia_out_signals(
        all_signals, all_memberships, target_model_idx, num_reference_models
    )
    mean_out_x = np.mean(out_signals, axis=1)
    mean_x = (1 + offline_a) / 2 * mean_out_x + (1 - offline_a) / 2

    # Compute log(p(x|θ)/p(x)) term
    log_ratio_x = np.log(target_signals.ravel() / mean_x)

    # Compute population signals for KL divergence
    z_signals = population_signals[:, target_model_idx]
    population_memberships = np.zeros_like(population_signals).astype(
        bool
    )  # All population data are OUT for all models
    z_out_signals = get_rmia_out_signals(
        population_signals,
        population_memberships,
        target_model_idx,
        num_reference_models,
    )
    mean_out_z = np.mean(z_out_signals, axis=1)
    mean_z = (1 + offline_a) / 2 * mean_out_z + (1 - offline_a) / 2

    # Compute D_KL(p(z) || p(z|θ)) term
    # p(z) is approximated by mean_z (population distribution)
    # p(z|θ) is the target model's prediction on population data z_signals
    # KL divergence: Σ p(z) * log(p(z) / p(z|θ))
    kl_divergence = np.mean(mean_z * np.log(mean_z / z_signals.ravel()))

    # Combine both terms: log(p(x|θ)/p(x)) + D_KL(p(z) || p(z|θ))
    test_statistic = log_ratio_x + kl_divergence

    return test_statistic
