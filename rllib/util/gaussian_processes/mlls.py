"""Implementation of MLL loss for multi-output GPs."""

import torch.nn as nn


def exact_mll(predicted_distribution, target, gp):
    """Calculate negative marginal log-likelihood of exact model."""
    data_size = target.shape[-1]
    loss = -predicted_distribution.log_prob(target).sum()

    if isinstance(gp, nn.ModuleList):
        for gp_ in gp:
            # Add additional terms (SGPR / learned inducing points,
            # heteroskedastic likelihood models)
            for added_loss_term in gp_.added_loss_terms():
                loss += added_loss_term.loss()

            # Add log probs of priors on the (functions of) parameters
            for _, prior, closure, _ in gp_.named_priors():
                loss += prior.log_prob(closure()).sum()
    else:
        for added_loss_term in gp.added_loss_terms():
            loss += added_loss_term.loss()

        # Add log probs of priors on the (functions of) parameters
        for _, prior, closure, _ in gp.named_priors():
            loss += prior.log_prob(closure()).sum()

    return loss / data_size
