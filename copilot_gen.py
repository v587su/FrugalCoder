from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import CBleu
from datasets import load_from_disk, Dataset
import datasets
import os
import random
import pickle
import re
random.seed(233)

python_codegen = load_from_disk('data/tosem/test/python/codegen')
python_codegen_scores = pickle.load(open('data/tosem/test/python_codegen_scores.pkl', 'rb'))
java_codegen = load_from_disk('data/tosem/test/java/codegen')
java_codegen_scores = pickle.load(open('data/tosem/test/java_codegen_scores.pkl', 'rb'))

python_starcoder = load_from_disk('data/tosem/test/python/starcoder')
python_starcoder_scores = pickle.load(open('data/tosem/test/python_starcoder_scores.pkl', 'rb'))

java_starcoder = load_from_disk('data/tosem/test/java/starcoder')
java_starcoder_scores = pickle.load(open('data/tosem/test/java_starcoder_scores.pkl', 'rb'))

codegen_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-7B", cache_dir='../cached')
starcoder_tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoderbase-7b', cache_dir='../cached')

def gen_annotations(dataset, scores, name='python', model='codegen'):
    ids = random.sample(range(len(dataset)), 383)
    print(ids)
    collection = []
    for i in ids:
        item = {
            'input_code': dataset['input_code'][i],
            'answer_code': dataset['answer_code'][i],
            'completed': scores['completed'][i],
            'bleu': scores['bleu'][i],
        }
        collection.append(item)
    os.makedirs(f'annotation/{name}_{model}', exist_ok=True)
    append = '.py' if name == 'python' else '.java'
    for i, item in enumerate(collection):
        with open(f'annotation/{name}_{model}/{i}{append}', 'w') as f:
            if name == 'java':
                code = re.sub(r'(?<=[\S])([ ]{4,})(?=[\S])', r'\n\1', item['input_code'])
                code = re.sub(r'(?<=[\S])([\t]{1,})(?=[\S])', r'\n\1', code)
            else:
                code = item['input_code']
            f.write(code)
            f.write('\n\n\n\n==================================\n\n')
            f.write(f'ANSWER: {item["answer_code"]}\n-------\n')
    

            if item['bleu'] == 1.0:
                f.write('\n\nMODEL EXACT MATCH')
            else:
                f.write(f'\n\nCOMPLETED: {item["completed"]}\n-------\n')

gen_annotations(python_codegen, python_codegen_scores, 'python', 'codegen')
gen_annotations(java_codegen, java_codegen_scores, 'java', 'codegen')
gen_annotations(python_starcoder, python_starcoder_scores, 'python', 'starcoder')
gen_annotations(java_starcoder, java_starcoder_scores, 'java', 'starcoder')

        


