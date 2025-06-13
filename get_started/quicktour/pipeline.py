from transformers import pipeline
import os 

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device="cpu")

result = pipeline("The secret to baking a good cake is ", max_length=50)
print(result)
