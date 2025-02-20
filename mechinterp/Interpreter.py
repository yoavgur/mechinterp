"""
This module defines the Interpreter class, which provides interpretation methods for transformer models.
"""
from contextlib import contextmanager
from typing import Iterator
import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer
from .ModelVector import InterpVector, LogitLensOutput, PatchscopesOutput, TunedLensOutput, VOProjectionOutput
from .internal_utils import InterpTensorType
from .utils import PatchscopesTargetPrompts
class Interpreter:
    """
    The Interpreter class is a wrapper around a HookedTransformer model that provides interpretation methods.

    Attributes:
        model (HookedTransformer): The transformer model to interpret.

    Methods:
        logit_lens(act: InterpTensorType, topk: int = 20, bottomk: int = 20, use_final_ln=True) -> LogitLensOutput:
            Run the logit lens on the given activations.
    """
    def __init__(self, model: HookedTransformer):
        """
        Initialize the Interpreter with a HookedTransformer model.
        """
        self.model = model

    def logit_lens(self, act: InterpTensorType, k: int = 20, use_final_ln=True, use_first_mlp=False) -> LogitLensOutput:
        """
        Run the logit lens on the given activations.
        """
        return InterpVector(self.model, act).logit_lens(k=k, use_final_ln=use_final_ln, use_first_mlp=use_first_mlp)
    
    def tuned_lens(self, act: InterpTensorType, l, k: int = 20) -> TunedLensOutput:
        return InterpVector(self.model, act).tuned_lens(l=l, k=k)
    
    def patchscopes(
            self,
            act: InterpTensorType,
            prompt: str = PatchscopesTargetPrompts.DESCRIPTION_FEW_SHOT,
            n: int = 30,
            target_model: HookedTransformer | None = None,
            target_layer: int = 2,
            temperature: float = 0.3,
            placeholder_token: str = "X",
            prepend_bos: bool = True
        ) -> PatchscopesOutput:
        """Apply patchscopes to the vector, using the given prompt.

        The vector will be patched into the placeholder positions in the prompt, like 'The meaning of {} is:'.
        If the vector is a batch, you must provide the same amount of placeholders as the batch size, and they'll be patched
        in to the corresponding positions, for example 'The meaning of {}{}{} is:' for a batch of size 3.

        Args:
            act: The vector to apply patchscopes to.
            prompt: Prompt to apply patchscopes to. The prompt must contain placeholders ('{}') where the vector will be
            patched in to. If the vector is a batch, you must provide the same amount of placeholders as the batch size,
            and they'll be patched in to the corresponding positions. The prompt defaults to a few-shot description prompt.
            n: Max number of tokens to generate.
            target_model: Model to apply patchscopes to - defaults to the model the object was initialized with.
            target_layer: Layer to apply patchscopes to - defaults to layer 2.
            temperature: Temperature for generation - defaults to 0.3.
            placeholder_token: Token to use for the placeholder - defaults to "X". This shouldn't matter unless the layer
            is very high, in which case it's possible that it'll start having some effect on next tokens.
            prepend_bos: Whether to prepend the BOS token to the prompt - defaults to True.

        Returns:
            PatchscopesOutput: an object containing the generated explanation.
        """

        return InterpVector(self.model, act).patchscopes(
            prompt=prompt,
            n=n,
            target_model=target_model,
            target_layer=target_layer,
            temperature=temperature,
            placeholder_token=placeholder_token,
            prepend_bos=prepend_bos
        )

    def vo_project(self, act: InterpTensorType, k: int = 20) -> VOProjectionOutput:
        raise NotImplementedError("VO projection not implemented yet")

    def activation_patching(self):
        raise NotImplementedError("Activation patching not implemented yet")
    
    def attribution_patching(self):
        raise NotImplementedError("Attribution patching not implemented yet")

    #### Maybe put under steering submodule? ####
    def diff_in_means(self, group1: list[str] | torch.Tensor, group2: list[str] | torch.Tensor) -> float:
        """
        Calculate the difference in means when doing forward passes with group1 vs group2.
        """
        # Can reference this - https://www.semanticscholar.org/reader/fe303bbaae47b1b08d0641b41d3288fcd74a3a80
        raise NotImplementedError("Diff in means not implemented yet")

    @contextmanager
    def activation_addition(self, act: Float[torch.Tensor, "d_model"], layer: int) -> Iterator[None]:
        """
        Adds hooks to steer the activations of the model.
        """
        raise NotImplementedError("Activation addition not implemented yet")

    @contextmanager
    def directional_ablation(self, act: Float[torch.Tensor, "d_model"], layer: int) -> Iterator[None]:
        """
        Adds hooks to ablate the activations of the model.
        """
        raise NotImplementedError("Directional ablation not implemented yet")
    
    #############################################
    