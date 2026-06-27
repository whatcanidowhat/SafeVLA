import torch
import json
import re
import os
import numpy as np
import random
import pdb
import math
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from transformers import AutoProcessor

from dataset_zoo import get_dataset
from misc import _default_collate
from model_zoo.llava import LlavaForConditionalGeneration, LlavaForConditionalGenerationScal
from model_zoo.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# MODEL = 'llava-hf/llava-1.5-7b-hf'
MODEL = "./data/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
root_dir = 'data'
DEVICE = "cuda:0" 
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
CHECKPOINT_DIR = "lprobe_checkpoint_qwen"
NUM_IMAGE_PATCHES = 576 

RELATION_TO_ID = {'on': 0, 'under': 1, 'left': 2, 'right': 3}
ID_TO_RELATION = {v: k for k, v in RELATION_TO_ID.items()}

# LABEL_MAPPING = {
#     'on': 0, 'above': 0, 
#     'under': 1, 'below': 1,
#     'left': 2, 
#     'right': 3
# }

SEED = 42
PROMPT_REGEX = re.compile(r"Where (?:is|are) the (.*?) in relation to the (.*?)\?", re.IGNORECASE)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_prompt(legacy_prompt):
    text = legacy_prompt.replace('<image>', '').strip()   
    if "USER:" in text:
        text = text.split("USER:")[-1].strip()  
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[0].strip()        
    return text

def run_evaluation(data_loader, model, processor, all_probe_heads, loss_function, num_layers):
    model.eval()
    all_probe_heads.eval()

    total_loss = 0
    total_items = 0
    
    probe_stats = {i: [0, 0] for i in range(num_layers)}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            i_option = batch["image_options"][0]
            index_of_total = batch["index"][0].item()
            img = i_option[0]

            original_prompt = prompt_list[index_of_total]
            relation_label = answer_list[index_of_total]
            # relation_id = LABEL_MAPPING.get(relation_label[0].lower())
            relation_id = RELATION_TO_ID.get(relation_label[0].lower())
            
            p_probe_prompt = clean_prompt(original_prompt)

            messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": img,
                                },
                                {"type": "text", "text": p_probe_prompt},
                            ],
                        }
                    ]

            text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
                                
            # Preprocess input for the model  
            single_input = processor(
                    text=[text], images=image_inputs, padding=True, return_tensors="pt"
            ).to(DEVICE)
 
            outputs = model(**single_input, output_hidden_states=True)    

            all_hidden_states = outputs.hidden_states 
            h_idx_End = -1

            labels_tensor = torch.tensor([relation_id], dtype=torch.long).to(DEVICE)
            
            batch_loss = 0
            for layer_idx in range(num_layers):
                current_layer_states = all_hidden_states[layer_idx + 1] 
                
                state_to_probe = current_layer_states.squeeze(0)[h_idx_End, :].to(torch.float32)
                
                logits = all_probe_heads[layer_idx](state_to_probe.unsqueeze(0))
                batch_loss += loss_function(logits, labels_tensor)
                
                pred = torch.argmax(logits, dim=1)
                
                probe_stats[layer_idx][1] += 1
                if pred.item() == relation_id:
                    probe_stats[layer_idx][0] += 1
                    
            total_loss += batch_loss.item()
            total_items += 1

    avg_loss = total_loss / (total_items * num_layers) if total_items > 0 else 0
    accuracies = {key: (probe_stats[key][0] / probe_stats[key][1]) * 100 if probe_stats[key][1] > 0 else 0 for key in probe_stats}
    
    return avg_loss, accuracies


def main():
    set_seed(SEED) 
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # TRAIN_DATASET = "COCO_QA_two_obj"
    # VAL_DATASET = "Controlled_Images_A"


    # 训练集加载
    # print(f"Loading Train Data: {TRAIN_DATASET}")
    # train_prompt_list = []
    # train_answer_list = []
    # with open(f'prompts/{TRAIN_DATASET}_with_answer_four_options.jsonl', 'r') as file:
    #     for line in file:
    #         data = json.loads(line)
    #         train_prompt_list.append(data["question"])
    #         train_answer_list.append(data["answer"])         
    # train_dataset = get_dataset(TRAIN_DATASET)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=_default_collate)

    # 验证集加载
    # print(f"Loading Val Data: {VAL_DATASET}")
    # val_prompt_list = []
    # val_answer_list = []
    # with open(f'prompts/{VAL_DATASET}_with_answer_four_options.jsonl', 'r') as file:
    #     for line in file:
    #         data = json.loads(line)
    #         val_prompt_list.append(data["question"])
    #         val_answer_list.append(data["answer"])         
    # val_dataset = get_dataset(VAL_DATASET)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=_default_collate)

    qst_ans_file = f'prompts/Controlled_Images_A_with_answer_four_options.jsonl'
    dataset = get_dataset("Controlled_Images_A")
    collate_fn = _default_collate 
    train_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    global prompt_list, answer_list
    prompt_list = []
    answer_list = []
    with open(qst_ans_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            prompt_list.append(data["question"])
            answer_list.append(data["answer"])

    global processor, model
    # model = LlavaForConditionalGenerationScal.from_pretrained(MODEL, revision='a272c74', cache_dir=root_dir, ignore_mismatched_sizes=True, local_files_only=True).to(DEVICE).eval()
    # processor = AutoProcessor.from_pretrained(MODEL, revision='a272c74', cache_dir=root_dir, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.float32,cache_dir=root_dir,ignore_mismatched_sizes=True,local_files_only=True).eval().to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL,cache_dir=root_dir,local_files_only=True)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # num_layers = model.language_model.config.num_hidden_layers
    num_layers = model.model.language_model.config.num_hidden_layers
    hidden_size = model.config.text_config.hidden_size
    
    all_probe_heads = nn.ModuleList(
        [nn.Linear(hidden_size, 4).to(DEVICE) for _ in range(num_layers)]
    )
    
    optimizer = optim.AdamW(all_probe_heads.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    
    best_layer_accuracies = [-1.0] * num_layers
    
    print("--- 开始L-Probing 训练 (32个独立探针 @ End token) ---")
    for epoch in range(NUM_EPOCHS):
        all_probe_heads.train()
        total_train_loss = 0
        items_processed = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            i_option = batch["image_options"][0]
            index_of_total = batch["index"][0].item()
            img = i_option[0]

            original_prompt = prompt_list[index_of_total]
            relation_label = answer_list[index_of_total]
            relation_id = RELATION_TO_ID.get(relation_label[0].lower())

            p_probe_prompt = clean_prompt(original_prompt)

            messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": img,
                                },
                                {"type": "text", "text": p_probe_prompt},
                            ],
                        }
                    ]

            text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
                                
            # Preprocess input for the model  
            single_input = processor(
                    text=[text], images=image_inputs, padding=True, return_tensors="pt"
            ).to(DEVICE)
            
            # p_probe_prompt = original_prompt
            # single_input = processor(text=p_probe_prompt, images=img, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad(): 
                outputs = model(**single_input, output_hidden_states=True)
            
            all_hidden_states = outputs.hidden_states 
            
            states_to_probe_len_check = all_hidden_states[-1].squeeze(0)
            input_ids = single_input["input_ids"][0]
            text_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

            # full_tokens_list = reconstruct_full_tokens(text_tokens, states_to_probe_len_check.shape[0])

            h_idx_End = -1
            
            labels_tensor = torch.tensor([relation_id], dtype=torch.long).to(DEVICE)
            
            total_loss = 0
            for layer_idx in range(num_layers):
                current_layer_states = all_hidden_states[layer_idx + 1]
                state_to_probe = current_layer_states.squeeze(0)[h_idx_End, :].to(torch.float32)
                
                logits = all_probe_heads[layer_idx](state_to_probe.unsqueeze(0))
                total_loss += loss_function(logits, labels_tensor)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_train_loss += total_loss.item()
            items_processed += 1
                
        avg_train_loss = total_train_loss / (items_processed * num_layers)

        # val_loss, val_accuracies = run_evaluation(
        #     val_loader, model, processor, all_probe_heads, loss_function, num_layers,
        #     val_prompt_list, val_answer_list
        # )
        
        val_loss, val_accuracies = run_evaluation(val_loader, model, processor, all_probe_heads, loss_function, num_layers)
        
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        has_new_best = False
        for layer_idx in range(num_layers):
            current_acc = val_accuracies[layer_idx]           
            if current_acc > best_layer_accuracies[layer_idx]:
                probe_file = f"probe_head_layer_{layer_idx}_best_new.pth"       
                if not has_new_best:
                    print("[Checkpoint]发现新的最佳探针:")
                print(f"Layer {layer_idx}: {current_acc:.2f}% (原最佳: {best_layer_accuracies[layer_idx]:.2f}%) -> 保存到 {probe_file}")               
                best_layer_accuracies[layer_idx] = current_acc
                probe_to_save = all_probe_heads[layer_idx]         
                torch.save(probe_to_save.state_dict(), os.path.join(CHECKPOINT_DIR, probe_file))              
                has_new_best = True
        if not has_new_best:
            print("(本轮没有探针创下新纪录)")

    print("\n---L-Probing 训练完成 ---")
            
    experiment_results = {
        "config": {
            "model": MODEL,
            "lr": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_dataset": "Controlled_A(10%)",
            "test_dataset": "Controlled_A(90%)",
            "seed": SEED
        },
        "results": {
            "layer_indices": list(range(num_layers)),
            "accuracies": best_layer_accuracies,
            "peak_layer": int(np.argmax(best_layer_accuracies)),
            "peak_accuracy": float(max(best_layer_accuracies))
        }
    }

    output_filename = "probing_id_qwen.json" 
    
    with open(output_filename, "w") as f:
        json.dump(experiment_results, f, indent=4)
        
    print(f"Data saved successfully to '{output_filename}'")
    print(f"Peak Accuracy: {max(best_layer_accuracies):.2f}% at Layer {np.argmax(best_layer_accuracies)}")
    
if __name__ == "__main__":
    main()