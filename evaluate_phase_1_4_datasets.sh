#!/bin/bash
#API_KEY=$1

# ===== KIE =====
# use '.' by default
main_data_dir="${1:-.}"


echo "Generating answers using expert for all data"
CUDA_VISIBLE_DEVICES=1 python eval_llava.py --output_path inference_results_new/output_4_datasets_all.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name converted_output_test.json \
--sub_ds_list docvqa,infographicvqa,websrc,wtq \
--main_data_dir MAIN_DATA_DIR


echo "Generating answers using expert for counting task"
CUDA_VISIBLE_DEVICES=1 python eval_llava.py --output_path inference_results_new/output_4_datasets_counting.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA_COUNTING \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name converted_output_test.json \
--sub_ds_list docvqa,infographicvqa,websrc,wtq \
--main_data_dir MAIN_DATA_DIR

echo "Generating answers using expert for non-counting task"
CUDA_VISIBLE_DEVICES=1 python eval_llava.py --output_path inference_results_new/output_4_datasets_non_counting.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA_NOT_COUNTING \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name converted_output_test.json \
--sub_ds_list docvqa,infographicvqa,websrc,wtq \
--main_data_dir MAIN_DATA_DIR



echo "Classifying task"
CUDA_VISIBLE_DEVICES=1 python classify_task.py --output_path inference_results_new/task_4_datasets.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name converted_output_test.json \
--sub_ds_list docvqa,infographicvqa,websrc,wtq \
--main_data_dir MAIN_DATA_DIR



echo "Merging experts"
python merge_experts.py --input_all_path inference_results_new/output_4_datasets_all.json \
--input_counting_path inference_results_new/output_4_datasets_counting.json \
--input_non_counting_path inference_results_new/output_4_datasets_non_counting.json \
--input_task_path inference_results_new/task_4_datasets.json \
--output_path inference_results_new/output_4_datasets_merged.json


