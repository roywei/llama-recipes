# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
from datasets import load_dataset
import torch
import lxml
import random
from dom_utils import get_tree_repr, prune_tree

# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] in targets:
            return True
    return False

def replace_target(target,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] == target:
            seq[i],seq[i+1],seq[i+2] = -100,-100,-100
    return seq

def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt,padding = True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i,n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs,current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq,labels)
        # Mask the padding token and image token 128256 
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256: #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


def get_custom_dataset(dataset_config, processor, split, split_ratio=0.9):
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset("osunlp/Multimodal-Mind2Web")

    dataset = dataset_dict['train']
    # Comment out the following line to use the full dataset, for quick testing only use 2000 samples
    dataset = dataset.select(range(2000))
    return dataset


def format_input_multichoice(
    sample, candidate_ids, gt=-1, previous_k=5, keep_html_brackets=False
):
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for idx, node in enumerate(candidate_nodes):
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )
    gt = id_mapping.get(gt, -1)
    seq_input = (
        "Based on the HTML webpage in the screenshot, try to complete the following task:\n"
        f"Task: {sample['confirmed_task']}\n"
        f"Previous actions:\n"
    )
    if len(sample["action_reprs"]) > 0 and sample['target_action_index'] > 0:
        # join previous actions before index
        seq_input += "\n".join(sample["action_reprs"][:sample['target_action_index']])
    else:
        seq_input += "None\n"
    seq_input += (
        "What should be the next action? Please select from the following choices "
        "(If the correct action is not in the page above, please select A. 'None of the above'):\n\n"
        "A. None of the above\n"
    )
    for idx, choice in enumerate(choices):
        # convert to ascii A, B, C, D, ...
        seq_input += f"{chr(66 + idx)}. {choice[1]}\n"
    if gt == -1:
        seq_target = "A."
    else:
        gt += 1
        current_action_op = sample["operation"]["op"]
        current_action_value = sample["operation"]["value"]
        seq_target = f"{chr(65 + gt)}.\n" f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"
    return tree_repr, seq_input, seq_target, choices

class Mind2WebDataCollator:
    def __init__(self, processor, num_candidates=10):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right
        self.num_candidates = num_candidates
    
    def __call__(self, samples):
        dialogs,images = [],[]
        for sample in samples:
            neg_candidates = sample["neg_candidates"]

            if len(sample["pos_candidates"]) != 0:
                
                pos_candidate = random.choice(sample["pos_candidates"])
                neg_candidate = random.sample(
                    neg_candidates,
                    min(len(neg_candidates), self.num_candidates - 1),
                )
                gt = pos_candidate["backend_node_id"]
                candidate_ids = [gt] + [c["backend_node_id"] for c in neg_candidate]
                seq_context, seq_in, seq_out, _ = format_input_multichoice(
                        sample, candidate_ids, gt
                    )
            else:
                # ground truth is None of the above
                neg_candidate = random.sample(
                    neg_candidates,
                    min(len(neg_candidates), self.num_candidates),
                )
                gt = -1
                candidate_ids = [c["backend_node_id"] for c in neg_candidate]
                seq_context, seq_in, seq_out, _ = format_input_multichoice(
                        sample, candidate_ids, gt
                    )

            try:
                image = sample["screenshot"].convert("RGB") 
            except:
                print("Error in converting image to RGB, skip this sample")
                continue

            dialog = [
                {"role":"user","content":[{"type": "image"},{"type": "text", "text": seq_in}]},
                {"role":"assistant","content":[{"type": "text", "text": seq_out}]}
            ]
            dialogs.append(dialog)
            images.append([image])
        
        return tokenize_dialogs(dialogs,images, self.processor)

def get_data_collator(processor):
    return Mind2WebDataCollator(processor)
