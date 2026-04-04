import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from ..mipha_arch import MiphaMetaModel, MiphaMetaForCausalLM


logger = logging.get_logger(__name__)


# 这里先用 AutoConfig 拿到实际 config 类型，避免一开始就把类写死
# 真正跑通后，再决定是否替换成更具体的 Qwen2_5_VLConfig / Qwen2Config
class MiphaQwenConfig(AutoConfig):
    model_type = "mipha_qwen"


class MiphaQwenModel(MiphaMetaModel, nn.Module):
    """
    Qwen backbone wrapper for SegEarth-R2.

    注意：
    这里不是复用 Qwen2.5-VL 的原生视觉入口，
    而是只复用其语言主干能力，继续吃我们自己构造的 inputs_embeds。
    """

    config_class = MiphaQwenConfig

    def __init__(self, config, qwen_model: Optional[nn.Module] = None):
        super(MiphaQwenModel, self).__init__(config)
        self.config = config

        if qwen_model is None:
            base_lm = AutoModelForCausalLM.from_config(config)
            if not hasattr(base_lm, "model"):
                raise AttributeError("Loaded model does not expose `.model`")
            qwen_model = base_lm.model
        self.qwen_model = qwen_model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


class MiphaQwenForCausalLM(nn.Module, MiphaMetaForCausalLM):
    """
    第一版目标：
    只保证接口和 mipha_phi.py 尽量一致，能让 llava_qwen.py 接上。
    """

    config_class = MiphaQwenConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, qwen_model: Optional[nn.Module] = None, lm_head: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        if qwen_model is None or lm_head is None:
            base_lm = AutoModelForCausalLM.from_config(config)
            if qwen_model is None:
                if not hasattr(base_lm, "model"):
                    raise AttributeError("Loaded model does not expose `.model`")
                qwen_model = base_lm.model
            if lm_head is None:
                if not hasattr(base_lm, "lm_head"):
                    raise AttributeError("Loaded model does not expose `.lm_head`")
                lm_head = base_lm.lm_head

        self.model = MiphaQwenModel(config, qwen_model=qwen_model)
        self.lm_head = lm_head

    def get_model(self):
        return self.model
    @property
    def device(self):
        return next(self.parameters()).device

    def post_init(self):
        # Keep interface compatible with HF PreTrainedModel-based code paths.
        return

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def get_input_embeddings(self):
        if hasattr(self.model.qwen_model, "embed_tokens"):
            return self.model.qwen_model.embed_tokens
        if hasattr(self.model.qwen_model, "model") and hasattr(self.model.qwen_model.model, "embed_tokens"):
            return self.model.qwen_model.model.embed_tokens
        raise AttributeError("Cannot find embed_tokens in qwen_model")

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        直接调用底层 Qwen 的 resize。
        """
        if hasattr(self.model.qwen_model, "resize_token_embeddings"):
            return self.model.qwen_model.resize_token_embeddings(new_num_tokens)
        raise AttributeError("Underlying qwen_model does not support resize_token_embeddings")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # 兼容 HF 常见输出
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            vocab_size = shift_logits.shape[-1]
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "images_clip": kwargs.get("images_clip", None),
                "seg_info": kwargs.get("seg_info", None),
                "class_name_embedding_indices": kwargs.get("class_name_embedding_indices", None),
                "class_name_ids": kwargs.get("class_name_ids", None),
                "cls_indices": kwargs.get("cls_indices", None),
                "dataset_type": kwargs.get("dataset_type", None),
            }
        )
        return model_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        第一版：直接从 HF 加载底层 Qwen CausalLM，再抽出 model/lm_head。
        后面如果确认 Qwen2.5-VL 具体类名，再收紧实现。
        """
        cache_dir = kwargs.pop("cache_dir", None)

        base_lm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            *model_args,
            **kwargs,
        )
        config = base_lm.config

        if hasattr(base_lm, "model"):
            qwen_model = base_lm.model
        else:
            raise AttributeError("Loaded model does not expose `.model`")

        if hasattr(base_lm, "lm_head"):
            lm_head = base_lm.lm_head
        else:
            raise AttributeError("Loaded model does not expose `.lm_head`")

        return cls(config=config, qwen_model=qwen_model, lm_head=lm_head)