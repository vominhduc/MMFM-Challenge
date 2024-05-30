#!/bin/bash
#API_KEY=$1

# ===== KIE =====
# use '.' by default
main_data_dir="${1:-.}"


echo "Generating answers using expert for all data"
CUDA_VISIBLE_DEVICES=0 python eval_llava_phase_2.py --output_path inference_results_new/output_mychart_all.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list mychart \
--main_data_dir MAIN_DATA_DIR


echo "Generating answers using expert for counting task"
CUDA_VISIBLE_DEVICES=0 python eval_llava_phase_2.py --output_path inference_results_new/output_mychart_counting.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA_COUNTING \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list mychart \
--main_data_dir MAIN_DATA_DIR


echo "Generating answers using expert for non-counting task"
CUDA_VISIBLE_DEVICES=0 python eval_llava_phase_2.py --output_path inference_results_new/output_mychart_non_counting.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA_NOT_COUNTING \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list mychart \
--main_data_dir MAIN_DATA_DIR


CUDA_VISIBLE_DEVICES=0 python classify_task_phase_2.py --output_path inference_results_new/task_mychart.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list mychart \
--main_data_dir MAIN_DATA_DIR


echo "Merging experts"
python merge_experts.py --input_all_path inference_results_new/output_mychart_all.json \
--input_counting_path inference_results_new/output_mychart_counting.json \
--input_non_counting_path inference_results_new/output_mychart_non_counting.json \
--input_task_path inference_results_new/task_mychart.json \
--output_path MAIN_DATA_DIR


echo "Generating answers using expert for all data"
CUDA_VISIBLE_DEVICES=0 python eval_llava_phase_2.py --output_path inference_results_new/output_mydoc_all.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list mydoc \
--main_data_dir MAIN_DATA_DIR


echo "Generating answers using expert for counting task"
CUDA_VISIBLE_DEVICES=0 python eval_llava_phase_2.py --output_path inference_results_new/output_mydoc_counting.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA_COUNTING \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list mydoc \
--main_data_dir MAIN_DATA_DIR


echo "Generating answers using expert for non-counting task"
CUDA_VISIBLE_DEVICES=0 python eval_llava_phase_2.py --output_path inference_results_new/output_mydoc_non_counting.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA_NOT_COUNTING \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list mydoc \
--main_data_dir MAIN_DATA_DIR


CUDA_VISIBLE_DEVICES=0 python classify_task_phase_2.py --output_path inference_results_new/task_mydoc.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list mydoc \
--main_data_dir MAIN_DATA_DIR


echo "Merging experts"
python merge_experts.py --input_all_path inference_results_new/output_mydoc_all.json \
--input_counting_path inference_results_new/output_mydoc_counting.json \
--input_non_counting_path inference_results_new/output_mydoc_non_counting.json \
--input_task_path inference_results_new/task_mydoc.json \
--output_path inference_results_new/output_mydoc_merged.json


echo "Generating answers using expert for all data"
CUDA_VISIBLE_DEVICES=0 python eval_llava_phase_2.py --output_path inference_results_new/output_myinfographic_all.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list myinfographic \
--main_data_dir MAIN_DATA_DIR


echo "Generating answers using expert for counting task"
CUDA_VISIBLE_DEVICES=0 python eval_llava_phase_2.py --output_path inference_results_new/output_myinfographic_counting.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA_COUNTING \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list myinfographic \
--main_data_dir MAIN_DATA_DIR


echo "Generating answers using expert for non-counting task"
CUDA_VISIBLE_DEVICES=0 python eval_llava_phase_2.py --output_path inference_results_new/output_myinfographic_non_counting.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA_NOT_COUNTING \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list myinfographic \
--main_data_dir MAIN_DATA_DIR



CUDA_VISIBLE_DEVICES=0 python classify_task_phase_2.py --output_path inference_results_new/task_myinfographic.json \
--data_path DATA_PATH \
--model_path PATH_TO_LORA \
--model_base lmsys/vicuna-13b-v1.5 --test_file_name annot_wo_answer.json \
--sub_ds_list myinfographic \
--main_data_dir MAIN_DATA_DIR


echo "Merging experts"
python merge_experts.py --input_all_path inference_results_new/output_myinfographic_all.json \
--input_counting_path inference_results_new/output_myinfographic_counting.json \
--input_non_counting_path inference_results_new/output_myinfographic_non_counting.json \
--input_task_path inference_results_new/task_myinfographic.json \
--output_path inference_results_new/output_myinfographic_merged.json