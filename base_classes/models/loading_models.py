from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype="auto",
    device_map="auto")


# Models and configurations


from transformers import AutoModelForCausalLM, MistralForCausalLM

# load with AutoClass or model-specific class
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype="auto", device_map="auto")
model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype="auto", device_map="auto")

# Model classes
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

# use the same API for 3 different tasks
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForQuestionAnswering.from_pretrained("meta-llama/Llama-2-7b-hf")

from transformers import AutoModelForCausalLM
# use the same API to load 3 different models
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")

# Large models
