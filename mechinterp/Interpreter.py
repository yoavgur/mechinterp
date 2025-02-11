"""
This module defines the Interpreter class, which provides interpretation methods for transformer models.
"""
from transformer_lens import HookedTransformer
from .ModelVector import InterpVector, LogitLensOutput, PatchscopesOutput, PSMappingFunction, TunedLensOutput
from .utils import InterpTensorType

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

    def tuned_lens(self, act: InterpTensorType, k: int = 20, lens: str | None = None) -> TunedLensOutput:
        return InterpVector(self.model, act).tuned_lens(k=k, lens=lens)
    
    def patchscopes(
            self,
            act: InterpTensorType,
            prompt: str,
            n: int = 20,
            target_token: str | None = None,
            target_position: int | None = None,
            mapping_function: PSMappingFunction = lambda x: x,
            target_model: HookedTransformer | None = None,
            target_layer: int = 1,
        ) -> PatchscopesOutput:
        return InterpVector(self.model, act).patchscopes(
            prompt=prompt,
            n=n,
            target_token=target_token,
            target_position=target_position,
            mapping_function=mapping_function,
            target_model=target_model,
            target_layer=target_layer
        )

    def activation_patching(self):
        raise NotImplementedError("Activation patching not implemented yet")
    
    def attribution_patching(self):
        raise NotImplementedError("Attribution patching not implemented yet")
    