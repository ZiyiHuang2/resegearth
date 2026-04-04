from .language_model.mipha_qwen import MiphaQwenForCausalLM, MiphaQwenModel
# from .language_model.mipha_gemma import MiphaGemmaForCausalLM
from .language_model.configuration_mipha import (
    MiphaVisionConfig,
    ProjectorConfig,
    MiphaPhiConfig,
)  # , MiphaGemmaConfig

# 为了向后兼容，创建Phi模型的别名
# 假设MiphaPhiForCausalLM和MiphaPhiModel是MiphaQwenForCausalLM和MiphaQwenModel的别名
MiphaPhiForCausalLM = MiphaQwenForCausalLM
MiphaPhiModel = MiphaQwenModel

# 可选：如果有单独的phi模型文件，应该从这里导入
# from .language_model.mipha_phi import MiphaPhiForCausalLM, MiphaPhiModel

__all__ = [
    'MiphaQwenForCausalLM',
    'MiphaQwenModel',
    'MiphaPhiForCausalLM',
    'MiphaPhiModel',
    'MiphaVisionConfig',
    'ProjectorConfig',
    'MiphaPhiConfig',
]