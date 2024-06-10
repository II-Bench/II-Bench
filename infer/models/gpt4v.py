from openai import OpenAI
import base64

API_KEY = ''
BASE_URL = ''

def load_model(model_path="GPT4V"):
    tokenizer = "GPT4V"
    model = "GPT4V"
    return tokenizer, model

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def make_content(text, image_paths):
    text_elem = {
        "type": "text",
        "text": text,
    }
    content = [text_elem]
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        image_elem = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low",
            },
        }
        content.append(image_elem)
    return content

def request(prompt, image_paths, timeout=60, max_tokens=1000):
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": make_content(prompt, image_paths),
            }
        ],
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response

def infer(tokenizer, model, text, images):
    try:
        response = request(text, images).choices[0].message.content
    except Exception as e:
        response = {"error": str(e)}
        print(response)
    return response
