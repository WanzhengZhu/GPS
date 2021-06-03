import argparse
import os
import torch
from tqdm import tqdm
from language_quality import extract_good_candidates_by_LQ
from utils import read_candidates, initialize_train_test_dataset, to_method_object, convert_to_contexts_responses


dir_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    print('Start Main...')
    ''' Read Candidates from Module 1 '''
    candidates = read_candidates('./data/' + args.dataset + '_candidates.txt')  # Load generated candidates from Module 1.
    train_x_text, train_y_text, test_x_text, test_y_text = initialize_train_test_dataset(args.dataset)
    contexts_train, responses_train = convert_to_contexts_responses(train_x_text, train_y_text)


    ''' Module 2: Candidates Pruning by Grammaticality '''
    candidates = extract_good_candidates_by_LQ(candidates, LQ_thres=0.52, num_of_generation=30000)
    method = to_method_object('TF_IDF')
    method.train(contexts_train, responses_train)
    good_candidates_index = method.sort_responses(test_x_text, candidates, min(args.kpq, len(candidates)))  # kpq: Top k candidates per query, for better computation.
    good_candidates = [[candidates[y] for y in x] for x in good_candidates_index]


    ''' Module 3: Response Selection '''
    METHODS = ['TF_IDF', 'BM25', 'USE_SIM', 'USE_MAP', 'USE_LARGE_SIM', 'USE_LARGE_MAP', 'ELMO_SIM', 'ELMO_MAP', 'BERT_SMALL_SIM', 'BERT_SMALL_MAP', 'BERT_LARGE_SIM', 'BERT_LARGE_MAP', 'USE_QA_SIM', 'USE_QA_MAP', 'CONVERT_SIM', 'CONVERT_MAP']
    for method_name in METHODS[14:16]:
        print(method_name)
        method = to_method_object(method_name)
        method.train(contexts_train, responses_train)
        output = []
        for i, test_i in enumerate(tqdm(test_x_text)):
            predictions = method.rank_responses([test_i], good_candidates[i])
            output.append(good_candidates[i][predictions.item()])
        print(output)
    print('*' * 80)
    print(f'After filtering by LQ, there are {len(candidates)} candidates.\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Main.py', description='choose dataset from reddit, gab, conan')
    parser.add_argument('--kpq', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='reddit', choices=['reddit', 'gab', 'conan'])
    args = parser.parse_args()
    main(args)

