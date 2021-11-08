import torch
from transformers import T5PreTrainedModel
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils import logging
logger = logging.get_logger(__name__)


class ParamEfficientTuning(T5PreTrainedModel):
    def __init__(self, config, pretrained_model, **kwargs):
        super().__init__(config)
        self.seq2seq_model = pretrained_model

        logger.info("Declare Paramter Efficient Tuning Model!")

        not_freeze_set = []
        if config.peft_option == 'bitfit':
            not_freeze_set = ['bias']
            all_match = True
        logger.info(not_freeze_set)

        for n, p in self.seq2seq_model.named_parameters():
            if config.peft_option == 'bitfit' and self.check_params(n, not_freeze_set, all_match=all_match):
                print("tune "+ n)
                p.requires_grad = True
            else:
                p.requires_grad = False

        logger.info("already freezed parameters!")

    def check_params(self, module_name, safe_list, all_match=True):
        check = [partial_name in module_name for partial_name in safe_list]
        return all(check) if all_match else any(check)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs,):

        # implement this method in case to add something here in the future
        output = self.seq2seq_model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask,
                                    head_mask=head_mask,
                                    decoder_head_mask=decoder_head_mask,
                                    cross_attn_head_mask=cross_attn_head_mask,
                                    encoder_outputs=encoder_outputs,
                                    past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds,
                                    decoder_inputs_embeds=decoder_inputs_embeds,
                                    labels=labels,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,
                                    **kwargs)
        return output
