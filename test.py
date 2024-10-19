import os
import gc
import torch
import argparse
import base64
from config import *
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from torch.utils.data import DataLoader
from eval.create_evaluator import Evaluator
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoProcessor, AutoModel, AutoTokenizer, LlavaForConditionalGeneration
from utils.utils import *
from datasets import load_dataset

   
def test(args):
    # Accelerator for DDP, FSDP, DeepSpeed, etc [Should First Call]
    accel = Accelerator()

    # 일단 llava만 불러오기
    if args.model == "llava":
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).cuda()
        processor = AutoProcessor.from_pretrained(model_id)
    elif args.model == "internVL2":
        path = "OpenGVLab/InternVL2-8B"
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif args.model == "IXC2b5":
        ckpt_path = "internlm/internlm-xcomposer2d5-7b"
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
        model.tokenizer = tokenizer
        model = model.eval()

    model.eval()

    # Initialize dataset & evaluator
    test_dataset = load_dataset("topyun/SPARK", split="train", cache_dir=args.dataset_dir)
    evaluator = Evaluator(root=args.dataset_dir)


    # Update dataset & evaluator
    evaluator.reset()
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=args.batch_size,
                                num_workers=4,
                                pin_memory=True,
                                collate_fn=lambda x: x)

    # Accel distributed
    test_dataloader = accel.prepare(test_dataloader)

    # progress bar
    prog_bar = tqdm(enumerate(test_dataloader), disable=not accel.is_local_main_process, total=len(test_dataloader))
    # eval start
    for batch_ind, inputs in prog_bar:

        # memory deallocation
        gc.collect()

        # removing cache
        torch.cuda.empty_cache()
        
        if args.model == "llava":
            all_predictions =[]
            for x in inputs:
                conversation = [
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": x['question_query']},
                        {"type": "image"},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

                raw_image = x['image'] 
                input = processor(prompt, raw_image, return_tensors='pt').to("cuda").to(torch.float16)

                output = model.generate(**input, max_new_tokens=64, do_sample=False)
                answer = processor.decode(output[0][2:], skip_special_tokens=True).split("ASSISTANT: ")[-1]
                all_predictions.append(answer)
        elif args.model == "internVL2":
            pixel_values = [load_image(x['image'], max_num=12).to(torch.bfloat16).cuda() for x in inputs]
            num_patches_list = [x.size(0) for x in pixel_values]
            pixel_values = torch.cat(pixel_values, dim = 0)
            questions = [x['question_query'] for x in inputs]
            
            generation_config = dict(max_new_tokens=1024, do_sample=False)
            
            responses = model.batch_chat(tokenizer, pixel_values,
                            num_patches_list=num_patches_list,
                            questions=questions,
                            generation_config=generation_config)
            all_predictions = responses
        elif args.model == "IXC2b5":
            all_predictions = []
            for x in inputs:
                query = '<ImageHere>'+x['question_query']
                image = [x['image']]
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    response, his = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
                all_predictions.append(response)
            
        for x in inputs: del x['image']
        evaluator.process(inputs, all_predictions)

        # garbage collection
        torch.cuda.empty_cache()
    
    print(f"[Device: {accel.device}] Finished!")
    accel.wait_for_everyone()
    # memory opt
    memory_optimization()

    # evaluate on dataset
    evaluator.evaluate(args.model, accel)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--model', default='llava', type=str, help='llava|internVL2|IXC2b5')
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()

    # test
    test(args)

