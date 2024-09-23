import copy
import torch
from torch import nn
from typing import Optional, Tuple, Union

from modeling_flash_t5 import FlashT5PreTrainedModel, FlashT5Stack, FlashT5CrossEntropyLoss, Seq2SeqLMOutput

from .config_flasht5lens import FlashT5Config


class FlashT5ForPretrainTasks(FlashT5PreTrainedModel):

    def __init__(self, config: FlashT5Config):
        super().__init__(config)
        config.is_encoder_decoder = False
        assert not config.tie_word_embeddings

        self.config = config
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = FlashT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = FlashT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.pop_mlp = nn.Linear(config.d_model, 10, bias=False)
        self.nml_mlp_link_layer = nn.Linear(config.d_model, 3, bias=False) # {No layer:0, Others:1, Ethernet:2}
        self.nml_mlp_network_layer = nn.Linear(config.d_model, 3, bias=False)   # {No Layer:0, Others:1, IP:2}
        self.nml_mlp_transport_layer = nn.Linear(config.d_model, 4, bias=False)  # {No Layer:0, Others:1, TCP:2, UDP:3}
        self.nml_mlp_app_layer = nn.Linear(config.d_model, 3, bias=False)  #{No Layer:0, Others:1, DNS:2}
        

        self.loss_fct = FlashT5CrossEntropyLoss(z_loss_factor=config.z_loss,
                                                label_smoothing=config.label_smoothing,
                                                use_triton_crossentropy=config.use_triton_crossentropy)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # do nothing
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        return model_inputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        max_length = 32,
        **kwargs,
    ) -> torch.LongTensor:
        """
            input_ids: B x L_encoder, int64
            attention_mask: B x L_encoder, int64
                1 for tokens to attend to, 0 for tokens to ignore

            Generation:
                Starts with 0, ends with 1, padding is 0

            # For 20 input/outputs, the diff between my implementation and HF is 9.8s vs 11.4s
        """
        B, _ = input_ids.size()
        labels = torch.zeros(B, 1, dtype=torch.long, device=input_ids.device)
        encoder_hidden_states = None

        for _ in range(max_length):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=labels,
                encoder_hidden_states=encoder_hidden_states,
            )
            encoder_hidden_states = out.encoder_hidden_states
            top_labels = out.logits[:, -1].argmax(-1).unsqueeze(-1)
            labels = torch.cat([labels, top_labels], dim=-1)

            if (labels == 1).sum(-1).clamp(min=0, max=1).sum().item() == B:
                break

        labels[:, -1] = 1

        # Mask out the padding, i.e., all positions after the first 1 with 0
        B, L = labels.size()
        mask = torch.arange(L, device=labels.device).unsqueeze(0) <= (labels == 1).long().argmax(-1).unsqueeze(-1)
        labels = labels.masked_fill(~mask, 0)

        return labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> Seq2SeqLMOutput:
        """
            input_ids: B x L_encoder, int64
            attention_mask: B x L_encoder, int64
                1 for tokens to attend to, 0 for tokens to ignore
            labels: B x L_decoder, int64
        """
        if encoder_hidden_states is None:
            encoder_hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]

        hidden_states = encoder_hidden_states

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )

        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss, z_loss = self.loss_fct(lm_logits, labels)
            loss += z_loss

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            encoder_hidden_states=encoder_hidden_states
        )