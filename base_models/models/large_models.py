# Fast initialization

# Sharded checkpoints
from transformers import AutoModel
import tempfile
import os

model = AutoModel.from_pretrained("biomistral/biomistral-7b")
with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="5GB")
    print(sorted(os.listdir(tmp_dir)))


with tempfile.TemporaryDirectory() as tmp_dir:
    # model.save_pretrained(tmp_dir)
    new_model = AutoModel.from_pretrained(tmp_dir)


from transformers.modeling_utils import load_sharded_checkpoint

with tempfile.TemporaryDirectory() as tmp_dir:
    # model.save_pretrained(tmp_dir, max_shard_size="5GB")
    load_sharded_checkpoint(model, tmp_dir)

import json 

with tempfile.TemporaryDirectory() as tmp_dir:
    with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "r") as f:
        index = json.load(f)

print(index.keys())
print(index['metadata'])
print(index['weight_map'])


# Big Model Inference
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")

device_map = {"model.layers.1": 0, "model.layers.14": 1, "model.layers.31": "cpu", "lm_head": "disk"}
model.hf_device_map

# Model data type
import torch
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype=torch.float16)

import torch
from transformers import AutoConfig, AutoModel

my_config = AutoConfig.from_pretrained("google/gemma-2b", torch_dtype=torch.float16)
model = AutoModel.from_config(my_config)