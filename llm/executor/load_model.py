# load qwen3-4B model from hugging face
# pass weights wo my qwen3 model

import torch

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from llm.model import Qwen3Config, Qwen3Model


class Qwen3Loader:
    def __init__(self, model_name="Qwen/Qwen3-4B"):
        assert model_name == "Qwen/Qwen3-4B", "only support Qwen/Qwen3-4B now"
        self.model_name = model_name

        self.config = self.load_config(model_name)
        self.tokenizer = self.load_tokenizer(model_name)
        self.model = self.load_model_weights(model_name)

    def _convert_qwen3_config(self):
        my_qwen3_config = Qwen3Config()
        my_qwen3_config.emb_out_hidden_size = self.config.hidden_size
        my_qwen3_config.vocab_size = self.config.vocab_size
        my_qwen3_config.num_layers = self.config.num_hidden_layers
        my_qwen3_config.q_num_head = self.config.num_attention_heads
        my_qwen3_config.kv_num_head = self.config.num_key_value_heads
        my_qwen3_config.head_dim = self.config.head_dim
        my_qwen3_config.theta = self.config.rope_theta
        my_qwen3_config.rms_eps = self.config.rms_norm_eps
        my_qwen3_config.max_seq_len = self.config.max_position_embeddings
        my_qwen3_config.use_tie_embedding = self.config.tie_word_embeddings
        return my_qwen3_config

    def _convert_qwen3_model(self, qwen3_config):
        my_model = Qwen3Model(config=qwen3_config)
        my_model.from_pretrained(self.model)
        return my_model

    def convert_official_model(self):
        qwen3_config = self._convert_qwen3_config()
        qwen3_model = self._convert_qwen3_model(qwen3_config)
        del self.model
        print("Convert QWen3 Model Success")
        return qwen3_model, qwen3_config, self.tokenizer

    def load_model_weights(self, model_name):
        official_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("Qwen3 Model Load Success")
        return official_model

    def load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Qwen3 Tokenizer Load Success")
        return tokenizer

    def load_config(self, model_name):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print("Qwen3 Config Load Success")
        return config
