from .slm_interface import SLM
import multiprocessing
from pytorch_pretrained_bert import BertForMaskedLM, tokenization
import torch
import numpy as np
from tqdm import tqdm
import logging
import sys
from .WSISettings import WSISettings

from typing import Dict, List, Tuple

from transformers import pipeline

# logger = logging.getLogger("bertwsi")


def get_batches(from_iter, group_size):
    ret = []
    for x in from_iter:
        ret.append(x)
        if len(ret) == group_size:
            yield ret
            ret = []
    if ret:
        yield ret

class LMBert(SLM):

    def __init__(self, cuda_device, bert_model, max_batch_size=20):
        super().__init__()
        logging.info(
            'creating bert in device %d. bert ath %s'
            ' max_batch_size: %d' % (
                cuda_device, bert_model,
                max_batch_size))

        device = torch.device("cuda")

        with torch.no_grad():
            model = BertForMaskedLM.from_pretrained(bert_model)
            model.cls.predictions = model.cls.predictions.transform
            model.to(device=device)
            model.eval()
            self.bert = model

            self.tokenizer = tokenization.BertTokenizer.from_pretrained(bert_model)

            self.max_sent_len = model.config.max_position_embeddings
            # self.max_sent_len = config.max_position_embeddings
            #
            self.max_batch_size = max_batch_size
            #
            self.lemmatized_vocab = []
            self.original_vocab = []

            import spacy
            nlp = spacy.load("en", disable=['ner', 'parser'])
            self._lemmas_cache = {}
            self._spacy = nlp
            for spacyed in tqdm(
                    nlp.pipe(self.tokenizer.vocab.keys(), batch_size=1000, n_threads=multiprocessing.cpu_count()),
                    total=len((self.tokenizer.vocab)), desc='lemmatizing vocab'):
                lemma = spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_
                self._lemmas_cache[spacyed[0].lower_] = lemma
                # if len(lemma) <= 1:  # or lemma in ['not', 'else', 'none', 'otherwise', 'something', 'like', 'whatev',
                #     # 'other', 'no', 'so', 'more',
                #     # 'even', 'exactly', 'really', 'yet', 'quite', 'much', 'actually',
                #     # 'anymore', 'entirely', 'necessarily', 'always', 'exclusively', 'only',
                #     # 'solely',
                #     # 'totally', 'precisely', 'just', 'surprisingly'
                #     # ]:  # or lemma.endswith('ly') :
                #     self.torch_out_bias[len(self.lemmatized_vocab)] = -1000000
                self.lemmatized_vocab.append(lemma)
                self.original_vocab.append(spacyed[0].lower_)

            self.device = device
    def format_sentence_to_pattern(self, pre, target, post, pattern):
        replacements = dict(pre=pre, target=target, post=post)
        for predicted_token in ['{mask_predict}', '{target_predict}']:
            if predicted_token in pattern: 
                before_pred, after_pred = pattern.split(predicted_token)
                before_pred = ['[CLS]'] + self.tokenizer.tokenize(before_pred.format(**replacements))
                after_pred = self.tokenizer.tokenize(after_pred.format(**replacements)) + ['[SEP]']
                target_prediction_idx = len(before_pred)
                target_tokens = ['[MASK]'] if predicted_token == '{mask_predict}' else self.tokenizer.tokenize(target)
                #print(before_pred + target_tokens + after_pred)
                #print(target_prediction_idx) 
                return before_pred + target_tokens + after_pred, target_prediction_idx
               
    def _get_lemma(self, word):
        if word in self._lemmas_cache:
            return self._lemmas_cache[word]
        else:
            spacyed = self._spacy(word)
            lemma = spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_
            self._lemmas_cache[word] = lemma
            return lemma

    def predict_sent_substitute_representatives_v2(self, unmasker, inst_id_to_sentence: Dict[str, Tuple[List[str], int]],
                                                wsisettings: WSISettings) \
            -> Dict[str, List[Dict[str, int]]]:
        """

        :param wsisettings: all algorithm settings
        :param inst_id_to_sentence: dictionary instance_id -> (sentence tokens list, target word index in tokens)

        """

        with torch.no_grad():
            
            '''
            print("\n\n1st and 2nd elements in inst_id_to_sentence before sorting by length")
            print(list(inst_id_to_sentence.items())[0])
            print(list(inst_id_to_sentence.items())[1])
            '''
            sorted_by_len = sorted(inst_id_to_sentence.items(), key=lambda x: len(x[1][0]) + len(x[1][2]))
            '''
            print("\n\n1st and 2nd elements in inst_id_to_sentence after sorting by length")
            print(sorted_by_len[0])
            print(sorted_by_len[1])
            '''
            
            res = {}
            for inst_id, (pre, target, post) in sorted_by_len:
                formatted_sent = pre + target + " (or even [MASK]) " + post
                #print("\n\n*****Formatted sentence: ****")
                #print(formatted_sent)
                reps = unmasker(formatted_sent, top_k= wsisettings.n_represents * wsisettings.n_samples_per_rep)
                #print("\n\nfirst represent: ")
                #print(reps[0])
                lemma = target.lower() if wsisettings.disable_lemmatization else self._get_lemma(target.lower())
                reps_list = [i['token_str'] for i in reps]
                
                new_reps = []
                for i in range(wsisettings.n_represents):
                    new_rep = {}
                    for j in range(wsisettings.n_samples_per_rep):
                        new_sample = reps_list.pop()
                        if new_sample == lemma:
                            continue
                        new_rep[new_sample] = 1  # rep.get(new_sample, 0) + 1
                    new_reps.append(new_rep)
                res[inst_id] = new_reps
            return res
    
    def predict_sent_substitute_representatives(self, inst_id_to_sentence: Dict[str, Tuple[List[str], int]],
                                                wsisettings: WSISettings) \
            -> Dict[str, List[Dict[str, int]]]:
        """

        :param wsisettings: all algorithm settings
        :param inst_id_to_sentence: dictionary instance_id -> (sentence tokens list, target word index in tokens)

        """
        # is_noun=False
        # for inst_id_temp in inst_id_to_sentence:
        #     if inst_id_temp.split('.')[1]=='n':
        #         is_noun=True
        #     break
        #
        # if is_noun:
        #     patterns=[('{pre} {target} (or even {mask_predict}) {post}', 0.5),
        #       ('{pre} {target_predict} {post}', 0.5)]
        # else:
        #     patterns =[('{pre} {target} (or even {mask_predict}) {post}', 0.5)]
        patterns = wsisettings.patterns
        n_patterns = len(patterns)
        pattern_str, pattern_w = list(zip(*patterns))
        pattern_w = torch.from_numpy(np.array(pattern_w, dtype=np.float32).reshape(-1, 1)).to(device=self.device)
        
        print("\n\npattern_str: ")
        print(pattern_str)
        print("pattern_w: ")
        print(pattern_w)
        
        with torch.no_grad():
            
            
            print("\n\n1st and 2nd elements in inst_id_to_sentence before sorting by length")
            print(list(inst_id_to_sentence.items())[0])
            print(list(inst_id_to_sentence.items())[1])
            
            sorted_by_len = sorted(inst_id_to_sentence.items(), key=lambda x: len(x[1][0]) + len(x[1][2]))
            
            print("\n\n1st and 2nd elements in inst_id_to_sentence after sorting by length")
            print(sorted_by_len[0])
            print(sorted_by_len[1])
            
            
            res = {}
            #print("\n\n# of batches " + str(len(list(get_batches(sorted_by_len,self.max_batch_size // n_patterns)))))
            for batch in get_batches(sorted_by_len,
                                     self.max_batch_size // n_patterns):

                batch_sents = []
                for inst_id, (pre, target, post) in batch:
                    #print("*****Formatted sentences: ****")
                    for pattern in pattern_str:
                        batch_sents.append(self.format_sentence_to_pattern(pre, target, post, pattern))
                
                print("\n\nSize of batch_sents (It has the sentences formatted based on pattern): "+str(len(batch_sents)))
                print("The first element in this list: ")
                print(batch_sents[0])
                print("The second element in this list: ")
                print(batch_sents[1])
                
                
                tokenized_sents_vocab_idx = [self.tokenizer.convert_tokens_to_ids(x[0]) for x in batch_sents]
                
                
                print("\n\nSize of tokenized_sents_vocab_idx : "+str(len(tokenized_sents_vocab_idx)))
                print("The first element in this list: ")
                print(tokenized_sents_vocab_idx[0])
                
                
                max_len = max(len(x) for x in tokenized_sents_vocab_idx)
                #print("\n\nmax_len: "+str(max_len))
                batch_input = np.zeros((len(tokenized_sents_vocab_idx), max_len), dtype=np.long)
                for idx, vals in enumerate(tokenized_sents_vocab_idx):
                    batch_input[idx, 0:len(vals)] = vals
                    
                
                print("Size of batch_input : "+str(batch_input.shape))
                print("The first element in this list: ")
                print(batch_input[0])
                
                torch_input_ids = torch.tensor(batch_input, dtype=torch.long).to(device=self.device)
                
                print("\n\nSize of torch_input_ids : "+str(torch_input_ids.shape))
                print("The first element in this list: ")
                print(torch_input_ids[0])
                
                torch_mask = torch_input_ids != 0
                
                print("\n\ntorch_mask"+ str(torch_mask.shape))
                print("torch_mask")
                print(torch_mask[0])
                

                logits_all_tokens = self.bert(torch_input_ids, attention_mask=torch_mask)
                
                print("\n\nSize of logits_all_tokens : "+str(logits_all_tokens.shape))
                print("The first element in this list: ")
                print(logits_all_tokens[0])
                
                
                logits_target_tokens = torch.zeros((len(batch_sents), logits_all_tokens.shape[2])).to(self.device)
                for i in range(0, len(batch_sents)):
                    logits_target_tokens[i, :] = logits_all_tokens[i, batch_sents[i][1], :]
                
                print("\n\nSize of logits_target_tokens : "+str(logits_target_tokens.shape))
                print("The first element in this list: ")
                print(logits_target_tokens[0])
                
                logits_target_tokens_joint_patt = torch.zeros(
                    (len(batch_sents) // n_patterns, logits_target_tokens.shape[1])).to(
                    self.device)
                
                for i in range(0, len(batch_sents), n_patterns):
                    
                    if i == 0:
                        print("\n\nTo see what does sum(0) do?")
                        print("without sum(0) " )
                        print(logits_target_tokens[i:i + n_patterns, :] * pattern_w)
                        print("with sum(0) ")
                        print((logits_target_tokens[i:i + n_patterns, :] * pattern_w).sum(0))
                        
                    logits_target_tokens_joint_patt[i // n_patterns, :] = (
                            logits_target_tokens[i:i + n_patterns, :] * pattern_w).sum(0)

                
                print("\n\nSize of logits_target_tokens_joint_patt : "+str(logits_target_tokens_joint_patt.shape))
                print("The first element in this list: ")
                print(logits_target_tokens_joint_patt[0])
                
                pre_softmax = torch.matmul(
                    logits_target_tokens_joint_patt,
                    self.bert.bert.embeddings.word_embeddings.weight.transpose(0, 1))

                topk_vals, topk_idxs = torch.topk(pre_softmax, wsisettings.prediction_cutoff, -1)

                probs_batch = torch.softmax(topk_vals, -1).detach().cpu().numpy()
                
                print("\n\nSize of probs_batch : "+str(probs_batch.shape))
                print("The first element in this list: ")
                print(probs_batch[0])
                
                
                topk_idxs_batch = topk_idxs.detach().cpu().numpy()
                
                print("\n\nSize of topk_idxs_batch : "+ str(topk_idxs_batch.shape))
                print("The first element in this list: ")
                print(topk_idxs_batch[0])
                
                
                for (inst_id, (pre, target, post)), probs, topk_idxs in zip(batch, probs_batch, topk_idxs_batch):
                    lemma = target.lower() if wsisettings.disable_lemmatization else self._get_lemma(target.lower())
                    logging.info(
                        f'instance {inst_id} sentence: {pre} --{target}-- {post}')
                    probs = probs.copy()
                    target_vocab = self.original_vocab if wsisettings.disable_lemmatization else self.lemmatized_vocab
                    
                    print("\n\nType of target_vocab : "+ str(type(target_vocab)))
                    print(target_vocab)

                    print("\n\nProbs")
                    
                    
                    for i in range(wsisettings.prediction_cutoff):
                            if target_vocab[topk_idxs[i]] == lemma:
                                #print(str(i) + target_vocab[topk_idxs[i]])
                                probs[i] = 0
                    probs /= np.sum(probs)

                    new_samples = list(
                        np.random.choice(topk_idxs, wsisettings.n_represents * wsisettings.n_samples_per_rep,
                                         p=probs))
                    
                    
                    print("\n\ntopk_idxs")
                    print(topk_idxs) # a list of numbers
                    print(topk_idxs.shape) # a tuple with size i guess
                    print(wsisettings.n_represents) # a number
                    print(wsisettings.n_samples_per_rep) # a number
                    
                    logging.info('some samples: %s' % [target_vocab[x] for x in new_samples[:5]])
                    
                    
                    print("\n\nSize of new_samples : "+ str(len(new_samples)))
                    print("The first element in this list: ")
                    print(new_samples[0])
                    

                    new_reps = []
                    for i in range(wsisettings.n_represents):
                        new_rep = {}
                        for j in range(wsisettings.n_samples_per_rep):
                            new_sample = target_vocab[new_samples.pop()] 
                            new_rep[new_sample] = 1  # rep.get(new_sample, 0) + 1
                        new_reps.append(new_rep)
                    res[inst_id] = new_reps
                    #sys.exit()

            return res

