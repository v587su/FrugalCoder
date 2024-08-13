from .dataset import BaseDataset
import random
random.seed(233)


class ScoredDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, is_dev=False, mode='train',max_pos_length=128, min_query_len=10, metric='bleu',model=None,language='java'):
        super().__init__(data_path, tokenizer, is_dev=is_dev, mode=mode, max_pos_length=max_pos_length, min_query_len=min_query_len, language=language)
        self.model = model
        self.metric = metric
        if self.mode == 'train':
            self.sequential('train', [align_method])
            self.sequential('test', [align_method])
        elif self.mode == 'eval':
            align_method = f'align_{self.model_type}_labels'
            self.sequential('test', [align_method])
   
    def align_labels(self, examples):
        examples['label'] = examples[self.metric]
        return examples

   


    
