import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib', 'MMMU', 'eval'))
# sys.path.append(os.path.join(os.path.dirname(__file__), 'lib', 'LLaVA'))

import torch
import random

import numpy as np
from tqdm import tqdm

import hashlib

import PIL
import datasets
import pandas as pd

from datasets import load_dataset, concatenate_datasets


from argparse import ArgumentParser

from utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from utils.model_utils import call_llava_engine_df, llava_image_processor
from utils.eval_utils import parse_multi_choice_response, parse_open_response

# sub_ds_list = ['llavar', 'docvqa', 'infographicsvqa', 'chartqa/val', 'scienceqa/val']
# test_file_name = 'converted_output_val.json'
shots = list(range(100)) # frozen set of shots to use, change for selecting specific indices

def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None,
              vis_process_func = None, vis_processors = None, device = None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            if sample['image']:
                sample['image'] = vis_process_func(sample['image'], vis_processors).to(device)
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)
            # release memory to avoid OOM
            sample['image'] = None
            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
            print(pred_ans)
    return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def conv2sample(args, convs, id, ds, image, _shots=None, load_image=True):
    ret = []
    for it in range(0, len(convs), 2):
        cur = {}
        assert convs[it]['from'] == 'human'
        cur['question'] = convs[it]['value'].replace('<image>\n', '').replace('<image>', '')
        cur['options'] = '[]'
        cur['explanation'] = ''
        if load_image:
            cur['image_1'] = PIL.Image.open(os.path.join(args.main_data_dir,  image))  # PIL.Image.open()  # args.data_path, ds,
        else:
            cur['image_1'] = os.path.join(args.main_data_dir, image) # args.data_path, ds,
        for jj in range(2, 8):
            cur[f'image_{jj}'] = None
        cur['img_type'] = f"['{ds}']"
        # assert convs[it + 1]['from'] == 'gpt'
        # cur['answer'] = convs[it + 1]['value']
        cur['answer'] = ''
        cur['topic_difficulty'] = 'Medium'
        cur['question_type'] = 'short-answer'
        cur['subfield'] = f'{ds}'

        # base_id = ds + '_' + str(id)
        # for_hash = base_id + ' ' + cur['question'] + ' ' + cur['answer']
        # suffix = hashlib.md5(str(for_hash).encode('utf-8')).hexdigest()
        cur['id'] = str(id) #+ '_' + suffix

        if _shots is not None:
            cur['shots'] = _shots

        ret.append(cur)
    return ret

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_13b_val.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="lib/MMMU/eval/configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="data/processed_data") # hf dataset path: MMMU/MMMU
    parser.add_argument('--main_data_dir', type=str, default=".") # hf dataset path: MMMU/MMMU
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--shots', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count_path', type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument('--moe_path', type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument('--model_base', type=str, default=None)

    parser.add_argument('--test_file_name', type=str, default='converted_output_val.json')
    parser.add_argument('--sub_ds_list', type=str, default='llavar,docvqa,infographicsvqa,chartqa/val,scienceqa/val')
    parser.add_argument('--eval_llava1_6', action='store_true', help='if llava1.6')
    parser.add_argument('--debug', action='store_true', help='enable debugger')

    args = parser.parse_args()

    if args.eval_llava1_6:
        # LLaVA 1.6 has different library than LLaVA 1.5
        sys.path.append(os.path.join(os.path.dirname(__file__), 'lib', 'LLaVA_1_6'))
    else:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'lib', 'LLaVA'))
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    if args.debug:
        from cvar_pyutils.debugging_tools import set_remote_debugger
        set_remote_debugger('9.67.169.241', 12345)

    print('llava_initializing...')
    processor = None
    call_model_engine = call_llava_engine_df
    vis_process_func = llava_image_processor

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    # for subject in CAT_SHORT2LONG.values():
    #     sub_dataset = load_dataset(args.data_path, subject, split=args.split)
    #     sub_dataset_list.append(sub_dataset)

    sub_ds_list = args.sub_ds_list.split(',')
    for ds in sub_ds_list:
        with open(os.path.join(args.data_path, ds, args.test_file_name), 'r') as f:
            ds_data = json.load(f)

        # few-shot support
        _shots = None
        if args.shots > 0:
            assert args.shots < len(shots)
            _shots = []
            for ix in range(args.shots):
                c = ds_data[shots[ix]]
                _shots.extend(conv2sample(args, c['conversations'], c['id'], ds, (c['image'] if 'image' in c else 'no_image.jpg'), load_image=False))

        tab_ds = []
        for c in tqdm(ds_data, desc=ds):
            if 'image' not in c:
                continue
            tab_ds.extend(conv2sample(args, c['conversations'], c['id'], ds, c['image'], _shots))
        sub_dataset = datasets.Dataset.from_list(tab_ds)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)


    # load base model
    model_name = get_model_name_from_path(args.model_path)
    print(model_name)
    tokenizer_moe, model_moe, vis_processors_moe, _ = load_pretrained_model(args.model_path, args.model_base, model_name)

    # # load count expert
    # model_name = get_model_name_from_path(args.count_path)
    # print(model_name)
    # tokenizer_count, model_count, vis_processors_count, _ = load_pretrained_model(args.count_path, args.model_base, model_name)

    # # load another expert
    # model_name = get_model_name_from_path(args.model_path)
    # print(model_name)
    # tokenizer_count, model_count, vis_processors_count, _ = load_pretrained_model(args.model_path, args.model_base, model_name)

    # exit()

    samples = []
    for sample in tqdm(dataset):
        # sample['question'] = f"Given an image, you are an expert to judge whether it is used for a counting task or not. Only answer Yes or No."
        sample['question'] = f"""You are an AI designed to classify tasks based on whether they involve counting objects or not. Tasks that involve filling icons in blanks, choosing icons, or questioning and answering on a textbook should be treated as counting tasks. Reading a form or receipt, and answering a question from a document image, an infographic image, a table, or a web source should be treated as non-counting tasks. Below are some examples:

            Example 1:
            Image: A picture of apples
            Question: "How many apples are there?"
            Task: Counting

            Example 2:
            Image: A picture of a car
            Question: "What color is the car?"
            Task: Non-counting

            Example 3:
            Image: A picture of people
            Question: "How many people are in the photo?"
            Task: Counting

            Example 4:
            Image: A picture of a cat
            Question: "Is the cat sleeping?"
            Task: Non-counting

            Example 5:
            Image: A picture of a multiple-choice question in a textbook with icons to choose from
            Question: "Choose the correct icon that matches the description."
            Task: Counting

            Example 6:
            Image: A blank diagram where you need to fill in the icons
            Question: "Fill in the blanks with the correct icons."
            Task: Counting

            Example 7:
            Image: A page from a textbook with a question and answer section
            Question: "Answer the questions on the textbook page."
            Task: Counting

            Example 8:
            Image: A picture of a receipt
            Question: "What is the total amount paid?"
            Task: Non-counting

            Example 9:
            Image: A form
            Question: "What is your name?"
            Task: Non-counting

            Example 10:
            Image: A table
            Question: "What is the average temperature for January?"
            Task: Non-counting

            Example 11:
            Image: An infographic image with a question
            Question: "What is the main idea conveyed in the infographic?"
            Task: Non-counting

            Example 12:
            Web Source: An article from a website with a question
            Question: "What is the author's main argument in the article?"
            Task: Non-counting

            Now, classify the following task:
            """ + "\n" + sample['question']

        # sample['question'] = f"""You are an AI designed to classify tasks based on whether they involve counting objects or not. Tasks that involve filling icons in blanks, choosing icons, or questioning and answering on a textbook should be treated as counting tasks. Reading a form or receipt, and answering a question from a document image, an infographic image, a table, or a web source should be treated as non-counting tasks. Below are some examples:
            
        #     Example 1:
        #     Image: A picture of apples
        #     Question: "How many apples are there?"
        #     Task: Counting

        #     Example 2:
        #     Image: A picture of a car
        #     Question: "What color is the car?"
        #     Task: Non-counting

        #     Example 3:
        #     Image: A picture of people
        #     Question: "How many people are in the photo?"
        #     Task: Counting

        #     Example 4:
        #     Image: A picture of a cat
        #     Question: "Is the cat sleeping?"
        #     Task: Non-counting   

        #     Example 5:
        #     Image: A picture of a multiple-choice question in a textbook with icons to choose from
        #     Question: "Choose the correct icon that matches the description."
        #     Task: Counting

        #     Example 6:
        #     Image: A webpage with information about a company's employees
        #     Question: "How many employees does Company Z have?"
        #     Task: Counting

        #     Example 7:
        #     Image: A page from a document with a title
        #     Question: "What is the title of the document?"
        #     Task: Non-counting
            
        #     Example 8:
        #     Image: A scanned form with fields for customer information
        #     Question: "What is the name of the customer?"
        #     Task: Non-counting

        #     Example 9:
        #     Image: A blank diagram where you need to fill in the icons
        #     Question: "Fill in the blanks with the correct icons."
        #     Task: Counting

        #     Example 10:
        #     Image: A page from a textbook with a question and answer section
        #     Question: "Answer the questions on the textbook page."
        #     Task: Non-counting

        #     Example 11:
        #     Image: An infographic displaying population data with a specific country highlighted
        #     Question: "What is the population of Country X?"
        #     Task: Counting
            
        #     Example 12:
        #     Image: A page from a textbook with numbered examples
        #     Question: "How many examples are provided in this chapter?"
        #     Task: Counting

        #     Example 13:
        #     Image: A grocery receipt with a list of purchased items
        #     Question: "What items were purchased in this transaction?"
        #     Task: Non-counting
            
        #     Example 14:
        #     Image: A webpage with information about a company's employee benefits
        #     Question: "What are the company's policies on employee benefits?"
        #     Task: Non-counting
            
        #     Example 15:
        #     Image: An article from a website with numbered references
        #     Question: "How many references are provided in the article?"
        #     Task: Counting
            
        #     Example 16:
        #     Image: A grocery receipt with a total amount section
        #     Question: "What is the total amount paid?"
        #     Task: Non-counting
            
        #     Now, classify the following task:
        #     """ + "\n" + sample['question']

        sample = process_single_sample(sample)
        

        sample = construct_prompt(sample, args.config)
        # if sample['image']:
        #     sample['image'] = vis_process_func(sample['image'], vis_processors).to(device)
        samples.append(sample)

    # run ex
    out_samples = run_model(args, samples, model_moe, call_model_engine, tokenizer_moe, processor, vis_process_func, vis_processors_moe, device)

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_json(args.output_path, out_samples)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()

