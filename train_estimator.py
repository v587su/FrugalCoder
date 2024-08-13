import datasets
from datasets import disable_caching
disable_caching()
import pickle
import os
import torch
import torch.utils.data as data
import numpy as np
import lightning as L
from model.lstm_estimator import LSTMClassifier

from sklearn import metrics
from transformers import BertForSequenceClassification, TrainingArguments, Trainer, GPT2Config, GPT2TokenizerFast, GPT2ForSequenceClassification, AutoTokenizer, BertConfig, EarlyStoppingCallback, DataCollatorWithPadding, AutoModelForCausalLM
from utils import arg_parser, load_dataset


from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfTransformer

def load_neural_model(args, checkpoint=None):
    if args.estimator == 'tcqe':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir='../cached')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.estimator == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', cache_dir='../cached')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif args.estimator == 'lstm':
        tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir='../cached')
        tokenizer.pad_token = tokenizer.eos_token
    # pretrained_model_name = 'bigcode/starcoderbase-7b' if args.model == 'starcoder' else 'Salesforce/codegen2-7B'
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, cache_dir='../cached')
    # tokenizer.model_max_length = 256
    # if args.model == 'starcoder':
    #     tokenizer.pad_token = tokenizer.bos_token
    #     tokenizer.padding_side = "left"
    # else:
    #     tokenizer.pad_token = tokenizer.eos_token
    # vocab_size = tokenizer.vocab_size if args.model == 'starcoder' else 51200
    if checkpoint is None:
        if args.estimator == 'tcqe':
            # model = GPT2ForSequenceClassification.from_pretrained('gpt2', cache_dir='../cached', num_labels=1, pad_token_id=tokenizer.pad_token_id)
            config = GPT2Config(n_embd=768, n_layer=1, n_head=4, num_labels=1, vocab_size=tokenizer.vocab_size, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            model = GPT2ForSequenceClassification(config).to(args.device)
            gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir='../cached')
            model.transformer.wte = gpt2_model.transformer.wte
            model.transformer.wpe = gpt2_model.transformer.wpe
        elif args.estimator == 'bert':
            config = BertConfig(intermediate_size=768, num_hidden_layers=1, num_attention_heads=4, num_labels=1, vocab_size=tokenizer.vocab_size)
            model = BertForSequenceClassification(config)
            model.config.pad_token_id = tokenizer.pad_token_id
            bert_model = AutoModelForCausalLM.from_pretrained('bert-base-cased', cache_dir='../cached')
            model.bert.embeddings.word_embeddings = bert_model.bert.embeddings.word_embeddings
            model.bert.embeddings.position_embeddings = bert_model.bert.embeddings.position_embeddings
            model.bert.embeddings.token_type_embeddings = bert_model.bert.embeddings.token_type_embeddings
        elif args.estimator == 'lstm':
            model = LSTMClassifier(batch_size=64, output_size=1, hidden_size=256, vocab_size=tokenizer.vocab_size, embedding_length=300, device=args.device)
            model.to(args.device)
    else:
        if args.estimator == 'tcqe':
            config = GPT2Config(n_embd=768, n_layer=1, n_head=4, num_labels=1, vocab_size=tokenizer.vocab_size, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            # config = GPT2Config(n_embd=768, n_layer=4, n_head=4, num_labels=1, vocab_size=tokenizer.vocab_size, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            model = GPT2ForSequenceClassification(config).to(args.device)
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
        elif args.estimator == 'bert':
            config = BertConfig(intermediate_size=768, num_hidden_layers=1, num_attention_heads=4, num_labels=1, vocab_size=tokenizer.vocab_size)
            # config = BertConfig(intermediate_size=768, num_hidden_layers=2, num_attention_heads=4, num_labels=1, vocab_size=tokenizer.vocab_size)
            model = BertForSequenceClassification(config).to(args.device)
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
        elif args.estimator == 'lstm':
            model = LSTMClassifier.load_from_checkpoint(checkpoint, batch_size=1, output_size=1, hidden_size=256, vocab_size=tokenizer.vocab_size, embedding_length=300, device=args.device).to(args.device)
            
    
    return model, tokenizer


def train_neural_model(args, train_dataset, test_dataset):
    model, tokenizer = load_neural_model(args)
    def code_to_ids(batch):
        tokenized = tokenizer(batch['input_code'], padding=True, truncation=True, max_length=args.text_length)
        labels = batch['label']
        encoded = {'input_ids': tokenized['input_ids'], 'label': labels, 'attention_mask': tokenized['attention_mask']}
        return encoded
    train_dataset = train_dataset.map(code_to_ids, batched=True)
    test_dataset = test_dataset.map(code_to_ids, batched=True)

    if args.estimator == 'lstm':
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=args.text_length)
        trainer = L.Trainer(
            default_root_dir=f'{args.output_dir}/{args.exp_name}/{args.language}_{args.model}_{args.estimator}_{args.metric}',
            max_epochs=20,
        )
        trainer.fit(
            model,
            data.DataLoader(train_dataset.with_format(type='torch',columns=['input_ids', 'label']), batch_size=64, shuffle=False, collate_fn=data_collator, drop_last=True), 
            data.DataLoader(test_dataset.with_format(type='torch',columns=['input_ids', 'label']), batch_size=64, shuffle=False, collate_fn=data_collator, drop_last=True), 
        )
    else:
        print('train begin')
        # print the number of parameters of the model
        print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=32)
        training_args = TrainingArguments(
            output_dir=f'{args.output_dir}/{args.exp_name}/{args.language}_{args.model}_{args.estimator}_{args.metric}',
            evaluation_strategy="epoch",
            eval_steps=1,
            save_strategy='epoch',
            load_best_model_at_end=True,
            learning_rate=5e-5,
            warmup_steps=1000,
            weight_decay=0.1,
            save_total_limit=10,
            num_train_epochs=args.epoch,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            # auto_find_batch_size=True, 64 batch size
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )
        trainer.train()
        # model.save_pretrained(os.path.join(args.output_dir, run_name, 'best_model'))
        model.save_pretrained(os.path.join(args.output_dir, args.exp_name, f'{args.language}_{args.model}_{args.estimator}_{args.metric}', 'best_model'))



def train_ml(args, train_dataset, test_dataset):
    train_dataset = train_dataset.with_format(type='numpy',columns=['input_ids','answers','input_code','answer_code'])
    test_dataset = test_dataset.with_format(type='numpy',columns=['input_ids','answers','input_code','answer_code'])
    X_train = train_dataset['input_code']
    X_test = test_dataset['input_code']
    y_train = train_dataset['label']
    y_test = test_dataset['label']

    if args.estimator == 'ada':
        rf_est = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', AdaBoostRegressor())], verbose=True)
    elif args.estimator == 'lr':
        rf_est = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', LinearRegression())], verbose=True)
    rf_est.fit(X_train, y_train)
    # save this model
    with open(os.path.join(args.output_dir, args.exp_name, f'{args.language}_{args.model}_{args.estimator}_{args.metric}.pickle'), 'wb') as f:
        pickle.dump(rf_est, f)
    
    predicted = rf_est.predict(X_test)
    print(metrics.mean_squared_error(y_test, predicted))
    print(metrics.r2_score(y_test, predicted))
    print(metrics.mean_absolute_error(y_test, predicted))

if __name__ == '__main__':
    args = arg_parser()
    print(args)
    train_dataset = load_dataset(args, 'train')
    test_dataset = load_dataset(args, 'test')
    # print the average length of the input code and the answer code
    print('average input code length:', np.mean([len(x) for x in train_dataset['input_code']]))
    print('average answer code length:', np.mean([len(x) for x in train_dataset['answer_code']]))
    # raise ValueError
    if args.estimator in ['tcqe', 'bert', 'lstm']:
        train_neural_model(args, train_dataset, test_dataset)
    elif args.estimator in ['lr', 'ada']:
        train_ml(args, train_dataset, test_dataset)
