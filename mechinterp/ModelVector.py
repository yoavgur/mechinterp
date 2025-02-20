"""Provides classes for interpreting model vectors, including logit lens functionality."""
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from .internal_utils import count_placeholders, format_toks, reshape_list, recursive_flatten, align_tensor, InterpTensorType, get_model_identifier
from .utils import PatchscopesTargetPrompts
from tuned_lens import TunedLens
from tuned_lens import load_artifacts
from huggingface_hub import list_models
from huggingface_hub import hf_hub_download


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

class VOProjectionOutput(LogitLensOutput):
    """Output of applying an attention head's VO to a token, and then projecting to vocabulary."""
    METHOD_NAME = "VO Projection"


class PatchscopesOutput:
    def __init__(self, explanation: str | list[str]):
        self.explanation = explanation

    def __str__(self):
        return f"Patchscopes Explanation: '{self.explanation}'"

    def __repr__(self):
        return self.__str__()

class InterpVector:
    """Represents a model vector and provides interpretation methods."""
    def __init__(self, model: HookedTransformer, vector: InterpTensorType):
        """Initialize InterpVector with a model and a vector."""
        self.model = model
        self.vector = align_tensor(vector, model.cfg.d_model).squeeze()

    def logit_lens(self, k: int = 20, use_final_ln=True, use_first_mlp=False) -> LogitLensOutput:
        """Perform logit lens analysis on the vector."""
        act = self.vector.clone()

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

    def tuned_lens(self, l, k: int = 20) -> TunedLensOutput:
        # Lenses can be taken from here - https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens

        model_name = self.model.cfg.model_name
        model_path = get_model_identifier(model_name)

        try:
            artifacts = load_artifacts.load_lens_artifacts(model_path)
        except:
            print("Model not found in tuned-lens repository")
            return

        with open(artifacts[1], 'rb') as f:
            lens = torch.load(f)
        
        weights = lens[f"{l}.weight"]
        bias = lens[f"{l}.bias"]

        act = self.vector.clone()
        device = act.device
        weights = weights.to(device)
        bias = bias.to(device)

        act = act + torch.matmul(act, weights.T) + bias

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

        return TunedLensOutput(topk_tokens, logits_topk.values, bottomk_tokens, logits_bottomk.values, act.shape[:-1], k)
    
    def vo_project(self, l, h, k: int = 20) -> VOProjectionOutput:
        """Apply the VO of an attention head to a token, and then project to vocabulary."""
        
        act = self.vector.clone()

        wv_matrix = self.model.W_V[l][h]
        wo_matrix = self.model.W_O[l][h]
        vo_matrix = torch.matmul(wv_matrix, wo_matrix)

        act = torch.matmul(act, vo_matrix)

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

        return VOProjectionOutput(topk_tokens, logits_topk.values, bottomk_tokens, logits_bottomk.values, act.shape[:-1], k)

    def patchscopes(
            self,
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

        target_model = target_model if target_model is not None else self.model

        vector = self.vector.clone()
        if len(vector.shape) == 1:
           vector = vector.unsqueeze(0)

        assert len(vector.shape) <= 2, f"Vector must be (d_model,) or (batch_size, d_model), got {vector.shape}"
        num_placeholders = count_placeholders(prompt)
        assert num_placeholders == vector.shape[0], f"Prompt must contain {vector.shape[0]} placeholders, got {num_placeholders}."

        prompt_toks, indices = format_toks(target_model, prompt, placeholder_token, prepend_bos=prepend_bos)

        # We define hook_ran since when using kv cache, the second time and onwards the hook is called with just one
        # token, i.e. we only need to apply the patch once.
        hook_ran = False
        def hook_patch_in_act(tensor: torch.Tensor, hook: HookPoint) -> torch.Tensor | None:
            nonlocal hook_ran
            if not hook_ran:
                for i in range(vector.shape[0]):
                    tensor[:, indices[i]] = vector[i]
                hook_ran = True

            return tensor

        with target_model.hooks(fwd_hooks=[(f"blocks.{target_layer}.hook_resid_pre", hook_patch_in_act)]):
            generated_toks = target_model.generate(prompt_toks.unsqueeze(0), max_new_tokens=n, verbose=False, temperature=temperature, use_past_kv_cache=True)

        return PatchscopesOutput(target_model.to_string(generated_toks)[0])
