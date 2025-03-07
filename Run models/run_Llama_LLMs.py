import torch
from unsloth import FastLanguageModel
import pandas as pd
import os

def generate_text(model, tokenizer, q_type, q):
    text = f"""Answer to the question below. 
        If the question is a Yes/No question (alt inv/ alt order), respond only with "yes" or "no". 
        If the question is a choice question (choice), respond only with choosing the group choice provided. 
        If the question starts with why (wh inv/ wh order), respond only with a short explanation starting with because.
        
        Question Type: {q_type}
        
        Question: {q}
        """
    FastLanguageModel.for_inference(model)

    messages = [
          {"role": "user", 
              "content": text},
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,return_tensors="pt").to("cuda")
    outputs = model.generate(inputs,  max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split('\n')[-1]
    return answer

max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token="hf_token"
)


def get_answers(input_path):
    
    dir_name = input_path.split('/')[2]
    file_name = input_path.split('/')[3]
    
    if not os.path.exists(f'Answers-8b-finetuned-full-1235/{dir_name}'):
        os.makedirs(f'Answers-8b-finetuned-full-1235/{dir_name}')
        
    df = pd.read_csv(input_path)

    for index, row in df.iterrows():
        print(index)
        question = row['question']
        q_type = row['type']
        answer  = generate_text(model, tokenizer, q_type, question)
        df.at[index, 'answer'] = answer


    df.to_csv(f"Answers-8b-finetuned-full-1235/{dir_name}/{file_name}", index=False)


get_answers("../sample_dataset/1-basic/basic_pair.csv")
get_answers("../sample_dataset/1-basic/basic_single.csv")
get_answers("../sample_dataset/2-MR1/MR1_pair.csv")
get_answers("../sample_dataset/2-MR1/MR1_single.csv")
get_answers("../sample_dataset/3-MR2/MR2_pair.csv")
get_answers("../sample_dataset/3-MR2/MR2_single.csv")
get_answers("../sample_dataset/10-MR1_somesome/pair_data.csv")
get_answers("../sample_dataset/10-MR1_somesome/single_data.csv")
get_answers("../sample_dataset/11-MR1_allall/MR1_all_all_pair.csv")
get_answers("../sample_dataset/11-MR1_allall/MR1_all_all_single.csv")
get_answers("../sample_dataset/12-MR1_someall/MR1_some_all_pair.csv")
get_answers("../sample_dataset/13-MR1_allsome/MR1_all_some_pair.csv")
get_answers("../sample_dataset/14-MR2_somesome/MR2_some_some_pair.csv")
get_answers("../sample_dataset/14-MR2_somesome/MR2_some_some_single.csv")
get_answers("../sample_dataset/15-MR2_all_all/MR2_all_all_pair.csv")
get_answers("../sample_dataset/15-MR2_all_all/MR2_all_single.csv")
get_answers("../sample_dataset/16-MR2_some_all/MR2_some_all.csv")
get_answers("../sample_dataset/17-MR2_all_some/MR2_all_some_pair.csv")
get_answers("../sample_dataset/18-MR1_MR2/MR1_MR2_pair.csv")
get_answers("../sample_dataset/18-MR1_MR2/MR1_MR2_single.csv")
get_answers("../sample_dataset/19-MR1_MR2_somesome/MR1_MR2_some_some_pair.csv")
get_answers("../sample_dataset/19-MR1_MR2_somesome/MR1_MR2_some_single.csv")
get_answers("../sample_dataset/20-MR1_MR2_allall/MR1_MR2_all_all_pair.csv")
get_answers("../sample_dataset/20-MR1_MR2_allall/MR1_MR2_all_single.csv")
get_answers("../sample_dataset/21-MR1_MR2_someall/MR1_MR2_some_all_pair.csv")
get_answers("../sample_dataset/22-MR1_MR2_allsome/MR1_MR2_all_some_pair.csv")
get_answers("../sample_dataset/23-MR3/MR3_pair.csv")
get_answers("../sample_dataset/23-MR3/MR3_single.csv")
get_answers("../sample_dataset/24-MR4/MR4_pair.csv")
get_answers("../sample_dataset/24-MR4/MR4_single.csv")
get_answers("../sample_dataset/25-MR5_allsome/MR5_allsome_pair.csv")
get_answers("../sample_dataset/26-MR5_someall/MR5_someall_pair.csv")
