import os
import gc
import torch
import argparse
import base64
import requests
from config import *
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from eval.create_evaluator import Evaluator
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datasets import load_dataset

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="jpeg")  # 이미지를 PNG 형식으로 저장 (다른 형식을 사용할 수도 있음)
    image_bytes = buffered.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")

def process_sample(model, input,count = 2):

    text = input['question_query']
    base64_image = encode_image(input['image'])
    max_tokens = 32
    temperature = 0
    if model == "gpt":    
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_KEY}"
        }
        payload = {
                "model": "gpt-4o", #select model
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": text
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{'jpeg'};base64,{base64_image}",
                            "detail": "low"
                        }
                        }
                    ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        done = False
        while count :
            try :
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                done = True
                break
            except :
                count -= 1
                time.sleep(1)
                
        return input, response.json()['choices'][0]['message']['content'].strip()
    elif model == "claude":
        headers = {
            "x-api-key": f"{CLAUDE_KEY}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
                "model": "claude-3-5-sonnet-20240620", #select model
                "messages": [
                    {"role": "user", "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": f"{base64_image}",
                            }
                        },
                        {"type": "text", "text": text}
                    ]}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        
        done = False
        while count :
            try :
                response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                done = True
                break
            except :
                count -= 1
                time.sleep(1)
        # print(response.json())
        return input, response.json()['content'][0]['text'].strip()
    elif model == "gemini":
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
                "contents": [
                    {"parts": [{
                        "text": text
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": f"{base64_image}"
                        }
                    }
                ]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 64
                }
            }
        
        done = False
        while count :
            try :
                response = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_KEY}", headers=headers, json=payload) #select model
                done = True
                break
            except :
                count -= 1
                time.sleep(1)
        # print(response.json())
        return input, response.json()['candidates'][0]["content"]["parts"][0]['text'].strip()


def test(args):
    accel = Accelerator()
    

    # Initialize dataset & evaluator
    test_dataset = load_dataset("topyun/SPARK", split="train")
    evaluator = Evaluator()
    results = {}
    
    evaluator.reset()
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=args.batch_size,
                                num_workers=4,
                                pin_memory=True,
                                collate_fn=lambda x: x)
    error_num = 0
    # progress bar
    prog_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    # eval start
    for batch_ind, inputs in prog_bar:
        all_predictions = []
        if not args.multiprocess:
            for input in inputs:
                all_predictions.append(process_sample(args.model,input))
        elif args.multiprocess:
            with ProcessPoolExecutor(max_workers=64) as executor:
                futures = [executor.submit(process_sample,args.model, input) for input in inputs]
                for future in as_completed(futures):
                    try:
                        if future.result():
                            all_predictions.append(future.result())
                    except:
                        pass
                    
        ids = [y[0]['id'] for y in all_predictions]
        for x in inputs:
            if x['id'] not in ids:
                all_predictions.append((x, 'Error'))
                error_num += 1
        inputs, all_predictions = zip(*all_predictions)
        prog_bar .set_description(f'{error_num}', refresh=True)
        evaluator.process(inputs, all_predictions)

    # evaluate on dataset
    evaluator.evaluate(args.model, accel)
    
    accel.print(results)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt', type=str, help='gpt|claude|gemini')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--multiprocess', default=False, type=bool)
    args = parser.parse_args()

    # test
    test(args)

