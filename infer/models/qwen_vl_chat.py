from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
torch.manual_seed(1234)

def load_model(model_path="Qwen/Qwen-VL-Chat"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
    return tokenizer, model

def infer(tokenizer, model, text, images):
    query = tokenizer.from_list_format([
        *[{'image': image} for image in images],
        {'text': text},
    ])
    # print(query)
    response, history = model.chat(tokenizer, query=query, history=None)
    # print(response)
    return response

