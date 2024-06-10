from .gpt4v import load_model as gpt4v_load_model, infer as gpt4v_infer
from .yi_vl_6b_chat import load_model as yi_vl_6b_chat_load_model, infer as yi_vl_6b_chat_infer
from .yi_vl_34b_chat import load_model as yi_vl_34b_chat_load_model, infer as yi_vl_34b_chat_infer
from .qwen_vl_chat import load_model as qwen_vl_chat_load_model, infer as qwen_vl_chat_infer

models = {
    'gpt4v': { # model name
        'load': gpt4v_load_model,
        'infer': gpt4v_infer,
        'model-path': 'GPT4V'
    },
    'yi-vl-6b-chat': {
        'load': yi_vl_6b_chat_load_model,
        'infer': yi_vl_6b_chat_infer,
        'model-path': '<your-model-path>'
    },
    'yi-vl-34b-chat': {
        'load': yi_vl_34b_chat_load_model,
        'infer': yi_vl_34b_chat_infer,
        'model-path': '<your-model-path>'
    },
    'qwen-vl-chat': {
        'load': qwen_vl_chat_load_model,
        'infer': qwen_vl_chat_infer,
        'model-path': 'Qwen/Qwen-VL-Chat'
    }
}

def load_model(choice):
    if choice in models:
        return models[choice]['load'](models[choice]['model-path'])
    else:
        raise ValueError(f"Model choice '{choice}' is not supported.")

def infer(choice):
    if choice in models:
        return models[choice]['infer']
    else:
        raise ValueError(f"Inference choice '{choice}' is not supported.")

