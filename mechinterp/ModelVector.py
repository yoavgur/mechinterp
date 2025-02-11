"""Provides classes for interpreting model vectors, including logit lens functionality."""
import torch
from typing import Callable
from transformer_lens import HookedTransformer
from .utils import reshape_list, recursive_flatten, align_tensor, InterpTensorType

PSMappingFunction = Callable[[torch.Tensor], torch.Tensor]
class LogitLensOutput:
    """Output of the logit lens, containing top and bottom tokens."""
    METHOD_NAME = "Logit Lens"

    def __init__(self, top_tokens: list[str], top_values: torch.Tensor, bottom_tokens: list[str], bottom_values: torch.Tensor, shape: tuple, k: int):
        """Initialize LogitLensOutput with top and bottom tokens."""
        self.t = self.top = top_tokens
        self.b = self.bottom = bottom_tokens

        self.tv = self.top_values = top_values
        self.bv = self.bottom_values = bottom_values

        self.shape = (*shape, k)

    def __str__(self):
        """Return a string representation of the LogitLensOutput."""
        return (
            f"{self.METHOD_NAME} Output:\n"
            f"\t- Topk tokens: {recursive_flatten(self.top)}\n\n"
            f"\t- Bottomk tokens: {recursive_flatten(self.bottom)}"
        )

    def __repr__(self):
        """Return a string representation of the LogitLensOutput."""
        return self.__str__()

class TunedLensOutput(LogitLensOutput):
    """Output of the tuned logit lens, containing top and bottom tokens."""
    METHOD_NAME = "Tuned Logit Lens"


class PatchscopesOutput:
    def __init__(self, explanation: str):
        self.explanation = explanation

    def __str__(self):
        return f"Patchscopes Explanation:\n{self.explanation}"

    def __repr__(self):
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
        logits_topk = torch.topk(logits, k, dim=-1, largest=True)
        logits_bottomk = torch.topk(logits, k, dim=-1, largest=False)

        topk_tokens = reshape_list(
            self.model.to_str_tokens(logits_topk.indices.flatten()),
            logits_topk.indices.shape
        )

        bottomk_tokens = reshape_list(
            self.model.to_str_tokens(logits_bottomk.indices.flatten()),
            logits_bottomk.indices.shape
        )

        return LogitLensOutput(topk_tokens, logits_topk.values, bottomk_tokens, logits_bottomk.values, act.shape[:-1], k)

    def tuned_lens(self, k: int = 20, lens: str | None = None) -> TunedLensOutput:
        # Lenses can be taken from here - https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens
        raise NotImplementedError("Tuned lens not implemented yet")

    def patchscopes(
            self,
            prompt: str,
            n: int = 20,
            target_token: str | None = None,
            target_position: int | None = None,
            mapping_function: PSMappingFunction = lambda x: x,
            target_model: HookedTransformer | None = None,
            target_layer: int = 1,
        ) -> PatchscopesOutput:
        # TODO: Assert that self.vector is the correct shape (d_mode). I guess we could support multi modes, but that
        # would mean doing so in a for loop, i.e. much less efficient than logit_lens.
        raise NotImplementedError("Patchscopes not implemented yet")
