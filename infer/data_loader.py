import json
import yaml

# Read the JSON data
def read_json(split='test'):
    with open(f'data/{split}.json', 'r') as json_file:
        return json.load(json_file)

# Read the YAML template
def read_yaml(config='none'):
    with open(f'config/{config}.yaml', 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

# Load the data
def load_data(split='test', mode='none', images_path='local_path'):
    samples = read_json(split)
    if mode == 'none' or mode == 'cot':
        config = mode
    elif mode == 'domain' or mode == 'emotion' or mode == 'rhetoric':
        config = 'keywords'
    elif mode == '1-shot' or mode == '2-shot' or mode == '3-shot':
        config = 'few-shot'
    else:
        raise ValueError(f"Invalid mode: {mode}")
    template = read_yaml(config)

    for sample in samples:
        questions = sample['questions']
        keywords = sample['meta_data']
        images = [sample[images_path]]
        
        prompts = []
        for question in questions:

            options = question['options']
            if mode == 'none' or mode == 'cot':
                prompt_format = [question['question'], *options]
                prompt = template['instruction'] + "\n" + template['prompt_format'][0].format(*prompt_format)

            elif mode == 'domain' or mode == 'emotion' or mode == 'rhetoric':
                prompt_format = [','.join([keywords[mode]] if not isinstance(keywords[mode], list) else keywords[mode]), question['question'], *options]
                prompt = template['instruction'] + "\n" + template['prompt_format'][0].format(*prompt_format)

            elif mode == '1-shot' or mode == '2-shot' or mode == '3-shot':
                shot_num = int(mode[0])
                prompt_format = [question['question'], shot_num+1, *options]
                prompt = f"{template['instruction'][shot_num-1]}" + '\n'.join([template['prompt_format'][i].format(i+1) for i in range(shot_num)]) + f"\n{template['prompt_format'][-1].format(*prompt_format)}"
                images = [image.strip() for image in template['few_shot_image'][0:shot_num]] + images
            else:
                raise ValueError(f"Invalid mode: {mode}")

            prompts.append(prompt)
        yield prompts, images, sample
        


if __name__ == '__main__':
    for prompt in load_data('test', '3-shot'):
        print(prompt)
        pass

