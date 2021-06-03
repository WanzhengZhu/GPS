import re
import string
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import glue_convert_examples_to_features
from transformers.data.processors.utils import InputExample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_candidates(candidates):
    for i in range(len(candidates)):
        candidates[i] = candidates[i].strip()
        candidates[i] = '. '.join(candidates[i].split('\n\n'))
        candidates[i] = '. '.join(candidates[i].split('\n'))
        candidates[i] = '.'.join(candidates[i].split('..'))
        candidates[i] = '. '.join(candidates[i].split('.'))
        candidates[i] = '. '.join(candidates[i].split('. . '))
        candidates[i] = '. '.join(candidates[i].split('.  . '))
        while len(candidates[i].split('  ')) > 1:
            candidates[i] = ' '.join(candidates[i].split('  '))

        myre = re.search(r'(\d+)\. (\d+)', candidates[i])
        while myre:
            candidates[i] = 'UNK'.join(candidates[i].split(myre.group()))
            myre = re.search(r'(\d+)\. (\d+)', candidates[i])
        if candidates[i] == "":
            candidates[i] = 'aaaaa'
        candidates[i] = candidates[i].strip()
    return candidates


def sent_tokenize_candidate(candidates):
    processed_candidates = []
    sen_length = []
    for candidate_i in candidates:
        temp = sent_tokenize(candidate_i)
        temp_len = 0
        for temp_i in temp:
            if len(temp_i.translate(str.maketrans('', '', string.punctuation)).split()) > 1:  # More than one word.
                processed_candidates.append(temp_i)
                temp_len += 1
        sen_length.append(temp_len)
    return processed_candidates, sen_length


def get_CoLA_score(candidates, model_name, saved_pretrained_CoLA_model_dir):
    def _load_pretrained_model(model_name, saved_pretrained_CoLA_model_dir):
        config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
        config = config_class.from_pretrained(saved_pretrained_CoLA_model_dir, num_labels=2, finetuning_task='CoLA')
        tokenizer = tokenizer_class.from_pretrained(saved_pretrained_CoLA_model_dir, do_lower_case=0)
        model = model_class.from_pretrained(saved_pretrained_CoLA_model_dir, from_tf=bool('.ckpt' in model_name), config=config).to(device)
        model.eval()
        return tokenizer, model

    def _evaluate(model, candidates, tokenizer, model_name):

        def __load_and_cache_examples(candidates, tokenizer):
            max_length = 128
            examples = [InputExample(guid=str(i), text_a=x) for i, x in enumerate(candidates)]
            features = glue_convert_examples_to_features(examples, tokenizer, label_list=["0", "1"], max_length=max_length, output_mode="classification")
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_labels = torch.tensor([0 for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([[0.0] * max_length for f in features], dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
            return dataset

        eval_dataset = __load_and_cache_examples(candidates, tokenizer)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, sampler=torch.utils.data.SequentialSampler(eval_dataset), batch_size=max(1, torch.cuda.device_count()))
        preds = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                if model_name.split('-')[0] != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if model_name.split('-')[0] in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        return preds[:, 1].tolist()

    tokenizer, model = _load_pretrained_model(model_name, saved_pretrained_CoLA_model_dir)
    temp_score = _evaluate(model, candidates, tokenizer, model_name)
    return [temp_score]


def convert_sentence_score_to_paragraph_score(temp_score, sen_length):
    paragraph_score = []
    for temp_i in temp_score:
        paragraph_score_i = []
        pointer = 0
        for i in sen_length:
            if i == 0:
                paragraph_score_i.append(0)
                continue
            temp_a = temp_i[pointer:pointer + i]
            paragraph_score_i.append(sum(temp_a) / len(temp_a))
            pointer += i
        paragraph_score.append(paragraph_score_i)
    return paragraph_score


def get_LQ_scores(candidates, model_name, saved_pretrained_CoLA_model_dir):
    candidates = preprocess_candidates(candidates)
    processed_candidates, sen_length = sent_tokenize_candidate(candidates)
    temp_score = get_CoLA_score(processed_candidates, model_name, saved_pretrained_CoLA_model_dir)
    temp_score = convert_sentence_score_to_paragraph_score(temp_score, sen_length)
    temp_score = [[max(0, y / 8.0 + 0.5) for y in x] for x in temp_score]  ## re-scale
    return temp_score[0]


def extract_good_candidates_by_LQ(candidates, LQ_thres, num_of_generation):
    model_name = 'bert-base-cased'
    saved_pretrained_CoLA_model_dir = './tmp/grammar_cola'
    to_test_candidates = candidates[:num_of_generation]
    LQ_scores = get_LQ_scores(to_test_candidates, model_name, saved_pretrained_CoLA_model_dir)
    scores = {i: j for i, j in zip(to_test_candidates, LQ_scores) if j > LQ_thres}
    good_candidates = list(scores.keys()) + candidates[num_of_generation:]
    good_candidates = list(set(good_candidates))
    return good_candidates

