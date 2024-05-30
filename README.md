## UPDATE
We use a Mixture of Experts in our approach, consisting of two steps: finetuning LLaVA 1.5 on different tasks to create multiple experts and selecting appropriate experts by using a pre-trained LLaVA 1.5.

Step 1: we observe that LLaVA 1.5 often fails on counting task. Therefore, we employ three experts in our approach including a counting expert, a non-counting expert, and a general expert. We then divide training datasets into three groups: (iconqa_fill, conqa_choose, textbookqa); 10 datasets excepting for (iconqa_fill, conqa_choose, textbookqa); and 10 datasets. Finally, we finetuned each expert on LLaVA 1.5 using LoRA.

Step 2: we first selecting expert by feeding the image and corresponding question into a pre-trained LLaVA 1.5 using the in-context learning with prompt as follows:

```
You are an AI designed to classify tasks based on whether they involve counting objects or not. Tasks that involve filling icons in blanks, choosing icons, or questioning and answering on a textbook should be treated as counting tasks. Reading a form or receipt, and answering a question from a document image, an infographic image, a table, or a web source should be treated as non-counting tasks. Below are some examples:

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
```

Then, depending on the selection, we use the corresponding expert to generate the answer. We note that due to the limitations of computing resources, we generate the answers using all experts first, and set a simple rule to merge the results using the MoE:

- If the task is counting: we will select the answer from the counting expert.

- If the task is not counting (in most cases, LLaVA also outputs the answer): we select the answer based on the consensus among 3 experts (at least 2/3 experts have the same answers). If no consensus, we will randomly choose between the non-counting expert and  the general expert since the counting expert is tailored only for the counting task.

### Train experts

We follow the authors to fine-tune LLaVA 1.5 13B instruction-tuned with LLAVA instruction-tuning data for each expert.

### Evaluation

run  `bash evaluate_phase_1_4_datasets.sh `  to generate answers for 4 datasets in phase 1

run  `bash evaluate_phase_1_4_datasets.sh `  to generate answers for 6 datasets in phase 1

run  `bash evaluate_phase_2.sh  ` to generate answers for datasets in phase 2

run  `python merge_json.py `  to merge all answer json files into one file following the format of the challenge

*Note: Please change PATH



## Overview
[MMFM Challenge](https://sites.google.com/view/2nd-mmfm-workshop/challenge?authuser=0).
Multimodal Foundation Models (MMFMs) have shown unprecedented performance in many computer vision tasks. However, on some very specific tasks like document understanding, their performance is still underwhelming. In order to evaluate and improve these strong multi-modal models for the task of document image understanding, we harness a large amount of publicly available and privately gathered data (listed in the image above) and propose a challenge. In the following, we list all the important details related to the challenge. Our challenge is running in two separate phases. For the first phase, we have released a training set consisting of publicly available data and for the second phase, an alien test set will be released soon.

This repository contains instructions for the challenge which is part of our 
[CVPR workshop on “What’s Next in Multimodal Foundation Models?”](https://sites.google.com/view/2nd-mmfm-workshop/home?authuser=0). 

## Table of Contents
1. [Important Dates](#important-dates)
2. [Prize](#prize)
3. [Getting Started](#instructions)
   1. [Installation](docs/installation.md)
   2. [Data Download](docs/download.md)
   3. [Evaluation](docs/evaluation.md)
   4. [Training](docs/training.md)
5. [Rules](#rules)
   - [General Rules](#general-rules)
   - [Challenge Phases](#challenge-phases)
   - [Metric](#metric)
6. [Submission](#submission)
7. [Baselines](#baselines)
8. [Disclaimer](#disclaimer)
9. [Contact](#contact)
10. [License](#license)


[//]: # (1. [Installation]&#40;docs/installation.md&#41;)

[//]: # (2. [Data Download]&#40;docs/download.md&#41;)

[//]: # (3. [Evaluation]&#40;docs/evaluation.md&#41;)

[//]: # (4. [Training]&#40;docs/training.md&#41;)


## Important Dates
- Phase 1 Data Release: March 20 2024
- Phase 2 Data Release: May 20 2024
- Final Evaluation Open: May 20 2024
- Phase 1&2 Due Date: 5th June 2024



## Prizes

<!-- The top 3 performers will be declared as the challenge winners and receive a prize totalling $10,000 USD, to be split as follows: -->

<!-- - 1st Place: $5000 -->
<!-- - 2nd Place: $3000 -->
<!-- - 3rd Place: $2000 -->


The challenge winners will be awarded with $10K in prizes sponsored by TensorLeap, and details about this will be shared later.



## Rules

### General Rules
- To be eligible for participation in our challenge, you must register your team via [MMFM Challenge CMT](https://cmt3.research.microsoft.com/MMFM2024).
    - Please select the `challenge track` while registering.

- The organizing committee reserves the right to verify the results and the code.
- For eligibility to win the prize, the participants are required to share their code and the model weights to the challenge. This is to ensure that the organizers can verify the results and the code. All submitted results would be documented in a leaderboard.

### Challenge Phases
- The challenge will be running in two phases: 
  - Phase 1: The participants will submit their results on the test sets which is already present in `data/pre_processed` for all datasets.
  - Phase 2: The participants will be required to submit their results on the alien test set. The alien test set will be of a similar domain to the Phase 1 `test data`.

### Submission
- For all submission, the participants will be required to submit results via [MMFM Online Evaluation](https://huggingface.co/spaces/ai-competition/MMFMChallenge) and submit an Instruction Document via [MMFM Challenge CMT](https://cmt3.research.microsoft.com/MMFM2024). We also ask teams to share the Github Repository to the reviewer account `mmfm2024` for repoducibility. 
    - Online Evaluation
       - One team use the same Huggingface account to login the online evaluation for all submissions
       - Please submit both Phase 1 and Phase 2 results
    - Instruction Document
       - The teamname and the Huggingface account name
       - 1-2 pages
       - The link of the Github repository with all necessary files or links for the repoducibility
       - The document should introduce how to reproduce the submitted results
       - The document should contain an overview of the methodology, training settings, and results for both phases
    - Github Repository
       - The repository should add the reviewer account `mmfm2024` as a repository member
       - The codes and Readme for the submission should be well-documented and easy to understand
       - The model weights (link to the accessible storage where the weights are uploaded)
       - The `requirements.txt` file containing the dependencies for the code

 ### Metric
  The evaluation metric is the overall score be computed as an average over the individual dataset results.


## Baselines
Here we provide three baselines for the challenge by training the LLAVA-1.5 model on three types of data, and we provide the results obtained from the 
resulting model on the test sets of the 10 datasets which contain 200 randomly sampled from the original `validation split` of the dataset. 
Due to the nature of the datasets, we evaluate the models with two metrics (and report the Accuracy %): 
- The [MMMU metric](https://arxiv.org/abs/2311.16502). Used for 6 datasetes: `iconqa_fill, funsd, iconqa_choose, wildreceipt, textbookqa, tabfact`
- Using Mixtral as a judge to evaluate the outputs of the models. Used for 4 datasets: `docvqa, inforgraphicvqa, websrc`


### Vanilla LLaVA Models:

- MMMU Evaluation 

| Model | Iconqa-Fill | Funsd | Iconqa-Choose | Wildreceipt | Textbookqa | Tabfact | Average |
|---------------|-------------|-------|---------------|-------------|------------|---------|---------|
| LLaVA 1.5 7B  | 13.5        | 21.5  | 36.0          | 6.0        | 37.5       | 54.0    | 28.1    |
| LLaVA 1.6 7B  | 13.0        | 17.5  | 38.5          | 20.0        | 44.0       | 49.0    | 30.3    |
| LLaVA 1.5 13B | 14.0        | 34.5  | 31.0          | 35.0        | 52.5       | 48.5    | 35.9    |
| LLaVA 1.6 13B | 14.5        | 39.5  | 35.0          | 44.5        | 54.5       | 47.5    | 39.3    |

- Mixtral Evaluation

| Model | DocVQA | InfographicsVQA | WebSRC | WTQ | Average |
|---------------|--------|-----------------|--------|-----|---------|
| LLaVA 1.5 7B  | 18.0   | 17.0            | 31.0   | 9.5 | 18.9    |
| LLaVA 1.6 7B  | 24.0   | 16.5            | 31.0   | 9.5 | 20.3    |
| LLaVA 1.5 13B | 22.5   | 20.5            | 29.5   | 8.0 | 20.1    |
| LLaVA 1.6 13B | 27.0   | 21.0            | 28.5   | 13.5 | 22.5    |



### LLaVA 1.5 13B instruction-tuned on the train sets of the 10 datasets:

- MMMU Evaluation 

| Iconqa-Fill | Funsd | Iconqa-Choose | Wildreceipt | Textbookqa | Tabfact | Average |
|-------------|-------|---------------|-------------|------------|---------|---------|
| 36.0        | 81.0  | 53.0          | 87.0        | 61.0       | 59.5    | 62.9    |

- Mixtral Evaluation

| DocVQA | InfographicsVQA | WebSRC | WTQ  | Average |
|--------|-----------------|--------|------|---------|
| 38.0   | 30.0            | 36.5   | 22.5 | 31.8    |

### LLaVA 1.5 13B instruction-tuned with LLAVA instruction-tuning data and the train sets of the 10 datasets:

- MMMU Evaluation 

| Iconqa-Fill | Funsd | Iconqa-Choose | Wildreceipt | Textbookqa | Tabfact | Average |
|-------------|-------|---------------|-------------|------------|---------|---------|
| 45.5        | 80.5  | 52.0          | 88.5        | 68.5       | 57.5    | 65.4    |

- Mixtral Evaluation

| DocVQA | InfographicsVQA | WebSRC | WTQ  | Average |
|--------|-----------------|--------|------|---------|
| 35.0   | 29.0            | 40.5   | 22.0 | 31.6    |

## Disclaimer
The organizers reserve the right to disqualify any participant who is found to be in violation of the rules of the challenge. 
The organizers also reserve the right to modify the rules of the challenge at any time.

## Contact
For any questions, please write an email to the organizers, a team member will get back to you as soon as possible:

[contactmmfm2024@gmail.com](contactmmfm2024@gmail.com).

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
To view the licenses of the datasets used in the challenge, please see [LICENSES](docs/licenses.md).
