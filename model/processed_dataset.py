import random
import os
import json
import pickle
from datasets import Dataset, load_from_disk, concatenate_datasets

class ProcessedDataset:
    def __init__(self, data_path, tokenizer, is_dev=False, mode='train',  max_pos_length=128, min_query_len=10, language='java'):
        self.data_path = data_path
        self.is_dev = is_dev
        self.tokenizer = tokenizer
        self.mode = mode
        self.language = language
        self.max_pos_length = max_pos_length
        self.min_query_len = min_query_len
        self.random_seed = 233
        random.seed(self.random_seed)

        self.dataset = self.load_data(data_path)
        self.sequential('train', [['tokenize_and_split','code']])
        self.sequential('test', [['tokenize_and_split','code']])

    def sequential(self, name, ops=[]):
        for op in ops:
            if isinstance(op, list):
                self.dataset[name] = self.dataset[name].map(lambda x: getattr(self, op[0])(x), batched=True, load_from_cache_file=False, remove_columns=op[1])
            else:
                self.dataset[name] = self.dataset[name].map(lambda x: getattr(self, op)(x), batched=True, load_from_cache_file=False)     

        
    def load_data(self, data_path):
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                lines = f.readlines()
                lines = lines[:1000] if self.is_dev else lines
                data = [json.loads(line) for line in lines]
            code =  [d['code'].strip() for d in data if len(d['code'].strip()) > 0]
            remove_dumplicated = sorted(list(set(code)))
            random.shuffle(remove_dumplicated)
            data = Dataset.from_dict({
                "code": remove_dumplicated,
            })
            data = data.train_test_split(test_size=0.1, seed=self.random_seed)
        elif 'final' in data_path and self.language == 'python':
            train_data = []
            test_data = []
            for file in os.listdir(os.path.join(data_path,'train')):
                if file.endswith('.jsonl'):
                    with open(os.path.join(data_path,'train', file), 'r') as f:
                        lines = f.readlines()
                    lines = lines[:100] if self.is_dev else lines
                    train_data += [json.loads(line) for line in lines]
                    if self.is_dev:
                        break
            for file in os.listdir(os.path.join(data_path,'test')):
                if file.endswith('.jsonl'):
                    with open(os.path.join(data_path, 'test',file), 'r') as f:
                        lines = f.readlines()
                    lines = lines[:100] if self.is_dev else lines
                    test_data += [json.loads(line) for line in lines]
            data = {
                'train': Dataset.from_dict({
                    "code": [d['code'].strip() for d in train_data if len(d['code'].strip()) > 0],
                }),
                'test': Dataset.from_dict({
                    "code": [d['code'].strip() for d in test_data if len(d['code'].strip()) > 0],
                }),
            }

        elif os.path.isdir(data_path) and not self.mode.startswith('eval'):
            shard_files = [name for name in os.listdir(data_path) if 'train_' in name]
            if len(shard_files) > 0:
                shards = [load_from_disk(os.path.join(data_path, f'train_{i}')) for i in range(len(shard_files))] 
                train_set = concatenate_datasets(shards)
                shards = [load_from_disk(os.path.join(data_path, f'test_{i}')) for i in range(len(shard_files))] 
                test_set = concatenate_datasets(shards)
            else:
                train_set = load_from_disk(os.path.join(data_path, 'train'))
                test_set = load_from_disk(os.path.join(data_path, 'test'))
            data = {
                'train': train_set,
                'test': test_set
            }
            if self.is_dev:
                data['train'] = data['train'].shard(num_shards=2000,index=0)
                data['test'] = data['test'].shard(num_shards=2000,index=0)
        else:
            print(data_path)
            raise ValueError

        return data



    def tokenize_and_split(self, examples):
        tokenized = self.tokenizer(examples['code'], max_length=self.max_pos_length+self.min_query_len, truncation=True)
        new_input_ids = []
        answers = []
        new_input_code = []
        answer_code = []
        split_num = 1

        for input_ids in tokenized['input_ids']:
            prompt_len = min(len(input_ids), self.max_pos_length)
            split_point = random.sample(range(1,prompt_len+1), split_num)
            for point in split_point:
                former_part = input_ids[:point]
                latter_part = input_ids[point : point + self.min_query_len]
                new_input_ids.append(former_part)
                new_input_code.append(self.tokenizer.decode(former_part, clean_up_tokenization_spaces=False))
                answers.append(latter_part)
                answer_code.append(self.tokenizer.decode(latter_part, clean_up_tokenization_spaces=False))
        
        return {'input_ids':new_input_ids, 'answers':answers, 'input_code': new_input_code, 'answer_code': answer_code}

            


        

        

        