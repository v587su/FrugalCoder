import os
import pickle
import re
import numpy as np
import torch
import tqdm
import datasets
from utils import arg_parser, CBleu
from model.processed_dataset import ProcessedDataset
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, GPT2ForSequenceClassification,CodeGenForCausalLM, AutoTokenizer

java_keywords = ['public ', '/**', 'protected ', 'private ', 'static ', 'class ', '@']
python_keywords = ['\ndef ',' def ', ' class ', '\nclass ', '@'] 

def compute_score(args, dataset, model, tokenizer, bleu, cbleu, name):
    stop_words = java_keywords if args.language == 'java' else python_keywords
    dataset = dataset.with_format(type='torch',columns=['input_ids','answers'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    eos_token_id = tokenizer.eos_token_id
    bleu_scores = []
    cbleu_scores = []
    completed_codes = []

    for batch in tqdm.tqdm(data_loader):
        answers = batch['answers'].numpy()[0].tolist()
        input_ids = batch['input_ids'].to(args.device)
        outputs = model.generate(input_ids, max_new_tokens=args.min_query_len, pad_token_id=eos_token_id, output_scores=True, return_dict_in_generate=True, min_length=0, do_sample=True, top_p=0.95, temperature=0.2)
        sequences = outputs['sequences'].cpu().numpy()
        output_seq = sequences[0][len(input_ids[0]):].tolist()
        if eos_token_id in output_seq:
            output_seq = output_seq[:output_seq.index(eos_token_id)]
        completed_str = tokenizer.decode(output_seq, clean_up_tokenization_spaces=False)
        for kw in stop_words:
            if kw in completed_str:
                new_complete = new_complete[:new_complete.index(kw)]
        new_complete = new_complete.rstrip()    

        if len(output_seq) == 0:
            bleu_scores.append({'bleu':0.0})
            cbleu_scores.append(0.0)
            completed_codes.append('<empty generated>')
            continue
       
        output_seq = tokenizer.encode(new_complete, add_special_tokens=False)
        if len(answers) == 0:
            bleu_scores.append({'bleu':0.0})
            cbleu_scores.append(0.0)
            completed_codes.append(completed_str)
            continue
        bleu_score = bleu.compute(predictions=[[str(i) for i in output_seq]], references=[[[str(i) for i in answers]]],smooth=True)
        bleu_scores.append(bleu_score)
        cbleu_scores.append(cbleu.sentence_bleu([str(i) for i in answers], [str(i) for i in output_seq]))
        completed_codes.append(completed_str)
        # print(f'input_code: {batch["input_code"][0]}\nanswer_code: {batch["answer_code"][0]}\ncompleted_code: {completed_str}\n')

    bleu_scores = [i['bleu'] for i in bleu_scores]
    scores = {
        'bleu': bleu_scores,
        'cbleu': cbleu_scores,
        'completed': completed_codes
    }
    file_name = f'{args.language}_{args.model}'
    if args.is_dev:
        file_name += '_dev'

    save_dir_path = os.path.join(args.data_dir, args.exp_name, name)
    os.makedirs(save_dir_path, exist_ok=True)
    with open(os.path.join(save_dir_path, f'{file_name}_scores.pkl'), 'wb') as f:
        pickle.dump(scores, f)
    dataset.save_to_disk(os.path.join(save_dir_path, args.language, args.model))


if __name__ == '__main__':
    args = arg_parser()
    if args.model == 'starcoder':
        tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoderbase-7b', cache_dir='../cached')
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained('bigcode/starcoderbase-7b', load_in_4bit=True, low_cpu_mem_usage=True, device_map='auto', cache_dir='../cached')
    elif args.model == 'gpt2':
        # deprecated
        raise ValueError('no longer supported')
        tokenizer = GPT2TokenizerFast.from_pretrained('codeparrot/codeparrot-small-multi')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained('codeparrot/codeparrot-small-multi').to(args.device)
    elif args.model == 'codegen':
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-7B", cache_dir='../cached')
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-7B", load_in_4bit=True, low_cpu_mem_usage=True, device_map='auto', cache_dir='../cached')
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    dataset = ProcessedDataset(os.path.join(args.data_dir, args.dataset_name), tokenizer, is_dev=args.is_dev, mode=args.mode, language=args.language, min_query_len=args.min_query_len, max_pos_length=args.text_length)


    bleu = datasets.load_metric('./cached/bleu/bleu.py')
    cbleu = CBleu([[str(i) for i in d] for d in dataset.dataset['train']['input_ids']])

    compute_score(args, dataset.dataset['train'], model, tokenizer, bleu, cbleu, 'train')
    compute_score(args, dataset.dataset['test'], model, tokenizer, bleu, cbleu, 'test')
    