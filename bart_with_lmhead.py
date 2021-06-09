from transformers import BartModel, BartPretrainedModel, BartConfig
import torch
from torch import nn
import os
from config import global_config as cfg

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class MyBart(BartPretrainedModel):
    def __init__(self, path, vocab_num):
        config = BartConfig.from_pretrained(path)
        super().__init__(config)
        self.model = BartModel.from_pretrained(path)
        self.register_buffer("final_logits_bias", torch.zeros((1, vocab_num)))
        if path != 'facebook/bart-base':
            self.lm_head = torch.load(os.path.join(path, 'lm_weight.json'))
        else:
            self.lm_head = nn.Linear(config.d_model, vocab_num, bias=False)

    def resize_token_embeddings(self, vocab_num):
        self.model.resize_token_embeddings(vocab_num)

    def forward(
        self,
        input=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        input_ids = input[:,0]
        labels = input[:,1]
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, cfg.pad_id, self.config.decoder_start_token_id
                )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        return [lm_logits]

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)
        lm_save_path = os.path.join(
            save_path, 'lm_weight.json')
        torch.save(self.lm_head, lm_save_path)
