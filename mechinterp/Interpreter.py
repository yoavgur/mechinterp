"""
This module defines the Interpreter class, which provides interpretation methods for transformer models.
"""
from transformer_lens import HookedTransformer
from .ModelVector import InterpVector, LogitLensOutput
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

    def logit_lens(self, act: InterpTensorType, topk: int = 20, bottomk: int = 20, use_final_ln=True) -> LogitLensOutput:
        """
        Run the logit lens on the given activations.
        """
        return InterpVector(self.model, act).logit_lens(topk=topk, bottomk=bottomk, use_final_ln=use_final_ln)