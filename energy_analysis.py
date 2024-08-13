import torch
import os

from deepspeed.profiling.flops_profiler import get_model_profile, FlopsProfiler
from transformers import  GPT2TokenizerFast, GPT2ForSequenceClassification,AutoModelForCausalLM,RobertaTokenizer, RobertaModel,AutoTokenizer, AutoModelForSeq2SeqLM, CodeGenForCausalLM, GPT2Config, BertConfig, BertForSequenceClassification


def input_constructor(batch_size, seq_len, tokenizer, name):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    if name in ['gpt2','codet5','gpt-neo']:
        # copy input_ids
        labels = inputs['input_ids'].clone()
    elif name in ['tcqe', 'bert']:
        labels = torch.tensor([1] * batch_size)
    else:
        labels = inputs['input_ids'].clone()

    inputs = dict(inputs)
    inputs.update({"labels": labels})
    
    for k, v in inputs.items():
        inputs[k] = v.to('cuda:0')
        # inputs[k] = v
    return inputs




targets = [{
    'path':'output/trained_models/tosem/java_codegen_tcqe_bleu/best_model',
    'model': GPT2ForSequenceClassification,
    'tokenizer':GPT2TokenizerFast.from_pretrained('gpt2', cache_dir='../cached'), 
    'name': 'tcqe'
    }, {
    'path':'output/trained_models/tosem/java_starcoder_bert_bleu/best_model',
    'model': BertForSequenceClassification,
    'tokenizer': AutoTokenizer.from_pretrained('bert-base-cased', cache_dir='../cached'),
    'name': 'bert'
    }]


# targets = [{'name': 'starcoder'}, {'name': 'tcqe'}, {'name': 'bert'} ]

for t in targets:
    # tokenizer = t['tokenizer']
    # tokenizer.pad_token = tokenizer.eos_token
    if t['name'] == 'tcqe':
        tokenizer = t['tokenizer']
        tokenizer.pad_token = tokenizer.eos_token
        config = GPT2Config(n_embd=768, n_layer=1, n_head=4, num_labels=1, vocab_size=tokenizer.vocab_size, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        # config = GPT2Config(n_embd=384, n_layer=4, n_head=4, num_labels=1, vocab_size=tokenizer.vocab_size, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        print(config.vocab_size)
        model = GPT2ForSequenceClassification(config).to('cuda:0')
        # model.load_state_dict(torch.load(os.path.join(t['path'], 'pytorch_model.bin')))
    elif t['name'] == 'bert':
        tokenizer = t['tokenizer']
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        config = BertConfig(num_hidden_layers=1, num_attention_heads=4, num_labels=1, vocab_size=tokenizer.vocab_size)
        # config = BertConfig(hidden_size=384, intermediate_size=768, num_hidden_layers=4, num_attention_heads=4, num_labels=1, vocab_size=tokenizer.vocab_size)
        print(f'bert vocab size: {config.vocab_size}')
        model = BertForSequenceClassification(config).to('cuda:0')
        # model.load_state_dict(torch.load(os.path.join(t['path'], 'pytorch_model.bin')))
    elif t['name'] == 'starcoder':
        tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoderbase-7b', cache_dir='../cached')
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained('bigcode/starcoderbase-7b', load_in_4bit=True, cache_dir='../cached')
        print(f'number of parameters: {model.num_parameters()}')
    elif t['name'] == 'codegen':
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-7B", cache_dir='../cached')
        model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen2-7B', load_in_4bit=True, cache_dir='../cached')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = t['model'].from_pretrained(t['path']).to('cuda:0')
    model.eval()
    batch_size = 1
    seq_len = 256
    name = t['name']
    input_shape = (batch_size, seq_len)
    prof = FlopsProfiler(model)
    inputs = input_constructor(batch_size, seq_len, tokenizer, name)
   

    for _ in range(1):
        if name in ['tcqe', 'bert']:
            _ = model(inputs['input_ids'])
        else:
            _ = model.generate(inputs['input_ids'], max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, eos_token_id=12345)
    exit = False
    for max_tk_num in [10,20,50,300]:
        print(name,max_tk_num)
        prof.start_profile(ignore_list=None)
        if name in ['tcqe', 'bert']:
            _ = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            exit = True
        else:
            _ = model.generate(inputs['input_ids'], max_new_tokens=max_tk_num, pad_token_id=tokenizer.eos_token_id, eos_token_id=12345)
            print(_)

        flops = prof.get_total_flops()
        macs = prof.get_total_macs()
        params = prof.get_total_params()
        prof.print_model_profile(profile_step=1,
                                module_depth=-1,
                                top_modules=1,
                                detailed=False,
                                output_file=None)
        prof.end_profile()
        if exit:
            break


    