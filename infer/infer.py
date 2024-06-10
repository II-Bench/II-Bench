from data_loader import load_data
from models import load_model, infer
import json
import sys
import argparse
from tqdm import tqdm
import os
import shutil

def check_completed(output_file):
    completed = {}
    no_response_id = []
    try:
        with open(output_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                for question in data['questions']:
                    if 'response' in question and (isinstance(question['response'], str) or (isinstance(question['response'], dict) and 'error' not in question['response'])):
                        completed[question['id']] = question['response']
                    else:
                        no_response_id.append(question['id'])
    except FileNotFoundError:
        pass  # 文件未找到时忽略
    except json.JSONDecodeError:
        pass  # JSON 解码错误时忽略
    return completed, no_response_id

def main(model_name='yi-vl-6b-chat', split='test', modes=['none', 'cot', 'domain', 'emotion', 'rhetoric', '1-shot', '2-shot', '3-shot'], output_dir='results', infer_limit=None):
    print('-'*100)
    print("[INFO] model_name:", model_name)
    print("[INFO] split:", split)
    print("[INFO] modes:", modes)
    print("[INFO] output_dir:", output_dir)
    print("[INFO] Infer Limit:", "No limit" if infer_limit is None else infer_limit)
    print('-'*100)
    tokenizer, model = load_model(model_name)
    os.makedirs(output_dir, exist_ok=True)
    for mode in modes:
        output_file_path = f'{output_dir}/{model_name}_{split}_{mode}.jsonl'
        temp_output_file_path = f'{output_file_path}.tmp'
        
        completed, _ = check_completed(output_file_path)
        temp_completed, _ = check_completed(temp_output_file_path)
        merged = {**temp_completed, **completed}
        infer_count = 0
        
        with open(temp_output_file_path, 'w') as temp_file:
            for prompts, image, sample in tqdm(load_data(split=split, mode=mode), desc=f'Processing {mode}'):
                for i, prompt in enumerate(prompts):
                    question = sample['questions'][i]
                    if question['id'] in merged:
                        question['prompt'] = prompt
                        question['response'] = merged[question['id']]
                        continue
                    if infer_limit is not None and infer_count >= infer_limit:
                        break
                    try:
                        response = infer(model_name)(tokenizer, model, prompt, image)
                        question['prompt'] = prompt
                        question['response'] = response
                        infer_count += 1
                    except Exception as e:
                        print(f"Error processing {prompt}: {e}", file=sys.stderr)
                        question['prompt'] = prompt
                        question['response'] = {"error": str(e)}

                json.dump(sample, temp_file)
                temp_file.write('\n')
                temp_file.flush()
                if infer_limit is not None and infer_count >= infer_limit:
                    break
        
        # Only rename the temp file to the final output file if the entire process completes successfully
        shutil.move(temp_output_file_path, output_file_path)
        _, no_response_id = check_completed(output_file_path)
        if len(no_response_id) > 0:
            print(f"Failed to get response for {len(no_response_id)} questions in {mode} mode. IDs: {no_response_id}", file=sys.stderr)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run inference and save results.')
    parser.add_argument('--model_name', type=str, default='yi-vl-6b-chat', help='Model name to use')
    parser.add_argument('--split', type=str, default='test', help='Data split to use')
    parser.add_argument('--mode', nargs='+', default=['none', 'cot', 'domain', 'emotion', 'rhetoric', '1-shot', '2-shot', '3-shot'], help='Modes to use for data loading, separated by space')
    parser.add_argument('--output_dir', type=str, default='results', help='File to write results')
    parser.add_argument('--infer_limit', type=int, help='Limit the number of inferences per run, default is no limit', default=None)
    args = parser.parse_args()

    main(model_name=args.model_name, split=args.split, modes=args.mode, output_dir=args.output_dir, infer_limit=args.infer_limit)
