# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" DistilBERT model configuration """
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import copy
import abc

import logging
from io import open

from configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'distilbert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-config.json",
    'distilbert-base-uncased-distilled-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-distilled-squad-config.json"
}

QUANT_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'distilbert-base-uncased': "https://www.dropbox.com/s/k7cdy59anau6x9v/config.json",  # noqa: E501
    'bert-large-uncased': "https://nlp-architect-data.s3-us-west-2.amazonaws.com/models/transformers/bert-large-uncased.json",  # noqa: E501
}


class DistilBertConfig(PretrainedConfig):
    pretrained_config_archive_map = DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 max_position_embeddings=512,
                 sinusoidal_pos_embds=False,
                 n_layers=6,
                 n_heads=12,
                 dim=768,
                 hidden_dim=4*768,
                 dropout=0.1,
                 attention_dropout=0.1,
                 activation='gelu',
                 initializer_range=0.02,
                 tie_weights_=True,
                 qa_dropout=0.1,
                 seq_classif_dropout=0.2,
                 **kwargs):
        super(DistilBertConfig, self).__init__(**kwargs)

        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.max_position_embeddings = max_position_embeddings
            self.sinusoidal_pos_embds = sinusoidal_pos_embds
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.dim = dim
            self.hidden_dim = hidden_dim
            self.dropout = dropout
            self.attention_dropout = attention_dropout
            self.activation = activation
            self.initializer_range = initializer_range
            self.tie_weights_ = tie_weights_
            self.qa_dropout = qa_dropout
            self.seq_classif_dropout = seq_classif_dropout
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")
    @property
    def hidden_size(self):
        return self.dim

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers

class QuantizedDistilBertConfig(DistilBertConfig):
    pretrained_config_archive_map = QUANT_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP



# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""
Generic config object:  
    load config from json file
    load config from ordinary python dict
    export config as dictionaty or json string
    define in init default parameters
"""
class QuantizationConfig(abc.ABC):
    """Quantization Configuration Object"""
    ATTRIBUTES = {
        "activation_bits": 8,
        "weight_bits": 8,
        "mode": "none",
        "start_step": 0,
        "ema_decay": 0.9999,
        "requantize_output": True
    }
    
    def __init__(self, **kwargs):
        for entry in self.ATTRIBUTES:
            setattr(self, entry, kwargs.pop(entry, self.ATTRIBUTES[entry]))
        if kwargs:
            raise TypeError(f"got an unexpected keyword argument: {list(kwargs.keys())}")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a config from a Python dictionary of parameters."""
        config = cls()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs Config from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        
        return cls.from_dict(json.loads(text))