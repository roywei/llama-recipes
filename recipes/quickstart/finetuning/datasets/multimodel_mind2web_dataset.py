import copy
from datasets import load_dataset
import itertools
import torch
from PIL import Image
import requests
from io import BytesIO

sys_prompt = '''Imagine that you are imitating humans doing web navigation for a task step by step. At each stage, you can see the webpage like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history. You need to decide on the first following action to take. You can click an element with the mouse, select an option, or type text with the keyboard. (For your understanding, they are like the click(), select_option() and type() functions in playwright respectively) One next step means one operation within the three.

The screenshot below shows the webpage you see. Follow the following guidance to think step by step before outlining the next action step at the current stage:

(Current Webpage Identification)
Firstly, think about what the current webpage is.

(Previous Action Analysis)
Secondly, combined with the screenshot, analyze each step of the previous action history and their intention one by one. Particularly, pay more attention to the last step, which may be more related to what you should do now as the next step.

(Screenshot Details Analysis)
Closely examine the screenshot to check the status of every part of the webpage to understand what you can operate with and what has been set or completed. You should closely examine the screenshot details to see what steps have been completed by previous actions even though you are given the textual previous actions. Because the textual history may not clearly and sufficiently record some effects of previous actions, you should closely evaluate the status of every part of the webpage to understand what you have done.

(Next Action Based on Webpage and Analysis)
Then, based on your analysis, in conjunction with human web browsing habits and the logic of web design, decide on the following action. And clearly outline which element in the webpage users will operate with as the first next target element, its detailed location, and the corresponding operation.

To be successful, it is important to follow the following rules: 
1. You should only issue a valid action given the current observation. 
2. You should only issue one action at a time'''

def check_header(targets, seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] in targets:
            return True
    return False

def replace_target(target, seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] == target:
            seq[i], seq[i+1], seq[i+2] = -100, -100, -100
    return seq

def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt, padding=True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs, current_seq):
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256:
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch

def get_custom_dataset(dataset_config, processor, split, split_ratio=0.9):
    dataset = load_dataset("osunlp/Multimodal-Mind2Web")
    dataset = dataset[split]
    return dataset


class Mind2WebDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, samples):
        dialogs, images = [], []
        for sample in samples:
            image = sample['screenshot']
            action_reprs = sample['action_reprs']
            target_index = int(sample['target_action_index'])
            target_action_reprs = sample['target_action_reprs']

            dialog = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{sys_prompt} Task: {sample['confirmed_task']}\nPrevious actions: {'; '.join(sample['action_reprs'])}"}
                ]},
                {"role": "assistant", "content": f"Based on the webpage screenshot and the given task, the next action should be:\n\nElement: {sample['target_action_reprs'].split(' -> ')[0]}\nOperation: {sample['operation']['op']}\nValue: {sample['operation']['value'] if sample['operation']['value'] else 'None'}"}
            ]
            dialogs.append(dialog)
            images.append([image])
        return tokenize_dialogs(dialogs, images, self.processor)

def get_data_collator(processor):
    return Mind2WebDataCollator(processor)

