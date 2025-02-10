"""Provides classes for interpreting model vectors, including logit lens functionality."""
from time import sleep
import torch
from typing import Union
from jaxtyping import Float
from transformer_lens import HookedTransformer
from .utils import reshape_list, recursive_flatten, align_tensor, InterpTensorType, transpose_tensor


class LogitLensOutput:
    """Output of the logit lens, containing top and bottom tokens."""
    def __init__(self, topk_tokens: list, bottomk_tokens: list, shape: tuple, k: int):
        """Initialize LogitLensOutput with top and bottom tokens."""
        self.topk = topk_tokens
        self.t = self.topk

        self.bottomk = bottomk_tokens
        self.b = self.bottomk

        self.shape = (*shape, k)

    def __str__(self):
        """Return a string representation of the LogitLensOutput."""
        return (
            "Logit Lens Output:\n"
            f"\t- Topk tokens: {recursive_flatten(self.topk)}\n\n"
            f"\t- Bottomk tokens: {recursive_flatten(self.bottomk)}"
        )

    def __repr__(self):
        """Return a string representation of the LogitLensOutput."""
        return self.__str__()

class InterpVector:
    """Represents a model vector and provides interpretation methods."""
    def __init__(self, model: HookedTransformer, vector: InterpTensorType):
        """Initialize InterpVector with a model and a vector."""
        self.model = model
        self.vector = align_tensor(vector, model.cfg.d_model)

    def logit_lens(self, k: int = 20, use_final_ln=True, use_first_mlp=False) -> LogitLensOutput:
        """Perform logit lens analysis on the vector."""
        act = self.vector.clone().squeeze()

        # TODO: Validate this with Amit
        if use_first_mlp:
            act = self.model.blocks[0].mlp(act)

        elif use_final_ln:
            act = self.model.ln_final(act)

        logits = self.model.unembed(act)
        topk_token_indices = torch.topk(logits, k, dim=-1, largest=True).indices
        bottomk_token_indices = torch.topk(logits, k, dim=-1, largest=False).indices

        topk_tokens = reshape_list(
            self.model.to_str_tokens(topk_token_indices.flatten()),
            topk_token_indices.shape
        )

        bottomk_tokens = reshape_list(
            self.model.to_str_tokens(bottomk_token_indices.flatten()),
            bottomk_token_indices.shape
        )

        return LogitLensOutput(topk_tokens, bottomk_tokens, act.shape[:-1], k)
