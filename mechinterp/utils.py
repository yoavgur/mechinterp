from enum import Enum

class PatchscopesTargetPrompts:
    IDENTITY_FEW_SHOT = "cat -> cat\n1135 -> 1135\nhello -> hello\n{}->"
    DESCRIPTION_FEW_SHOT = "Syria: Country in the Middle East\nLeonardo DiCaprio: American actor\nSamsung: South Korean multinational major appliance and consumer electronics corporation\n{}:"

class PatchingMethod(Enum):
    ResidAttnMLP = "resid_attn_mlp"
    ResidPre = "resid_pre"
    ResidMid = "resid_mid"
    MlpOut = "mlp_out"
    AttnOut = "attn_out"
    AttnHead = "attn_head"
    AttnHeadAllPos = "attn_head_all_pos"