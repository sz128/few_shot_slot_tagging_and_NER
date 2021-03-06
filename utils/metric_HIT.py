
import sys

def compute_fscore(TP, FP, FN):
    metrics = {'p': 0, 'r': 0, 'f': 0}
    if TP + FP == 0:
        metrics['p'] = 'Nan'
    else:
        metrics['p'] = 100 * TP / (TP + FP)
    if TP + FN == 0:
        metrics['r'] = 'Nan'
    else:
        metrics['r'] = 100 * TP / (TP + FN)
    if 2 * TP + FP + FN == 0:
        metrics['f'] = 'Nan'
    else:
        metrics['f'] = 100 * 2 * TP / (2 * TP + FP + FN)
    return metrics

def analysis_fscore(pred_items, label_items):
    """
    pred_items: a set
    label_items: a set
    """
    TP, FP, FN = 0, 0, 0
    for pred_item in pred_items:
        if pred_item in label_items:
            TP += 1
        else:
            FP += 1
    for label_item in label_items:
        if label_item not in pred_items:
            FN += 1
    return TP, FP, FN

def get_tuples_of_slot_and_intent(words, chunks, intents):
    tuples = set()
    for start_idx, end_idx, slot_name in chunks:
        tuples.add((slot_name, ''.join(words[start_idx - 1:end_idx]).replace(' ', '')))
    for intent in intents:
        if intent not in {'<pad>', '<unk>', '<EMPTY>'}:
            items = intent.split('-')
            if len(items) < 3:
                tuples.add(tuple(items))
            else:
                act, slot, value = items
                tuples.add((act + '-' + slot, value.replace(' ', '')))
    return tuples

def get_chunks(labels):
    """
        It supports IOB2 or IOBES tagging scheme.
        You may also want to try https://github.com/sighsmile/conlleval.
    """
    chunks = []
    start_idx,end_idx = 0,0
    for idx in range(1,len(labels)-1):
        chunkStart, chunkEnd = False,False
        if labels[idx-1] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            prevTag, prevType = labels[idx-1][:1], labels[idx-1][2:]
        else:
            prevTag, prevType = 'O', 'O'
        if labels[idx] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            Tag, Type = labels[idx][:1], labels[idx][2:]
        else:
            Tag, Type = 'O', 'O'
        if labels[idx+1] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            nextTag, nextType = labels[idx+1][:1], labels[idx+1][2:]
        else:
            nextTag, nextType = 'O', 'O'

        if Tag == 'B' or Tag == 'S' or (prevTag, Tag) in {('O', 'I'), ('O', 'E'), ('E', 'I'), ('E', 'E'), ('S', 'I'), ('S', 'E')}:
            chunkStart = True
        if Tag != 'O' and prevType != Type:
            chunkStart = True

        if Tag == 'E' or Tag == 'S' or (Tag, nextTag) in {('B', 'B'), ('B', 'O'), ('B', 'S'), ('I', 'B'), ('I', 'O'), ('I', 'S')}:
            chunkEnd = True
        if Tag != 'O' and Type != nextType:
            chunkEnd = True

        if chunkStart:
            start_idx = idx
        if chunkEnd:
            end_idx = idx
            chunks.append((start_idx,end_idx,Type))
            start_idx,end_idx = 0,0
    return chunks

if __name__=='__main__':
    import argparse
    import prettytable
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', required=True, help='path to dataset')
    parser.add_argument('-p', '--print_log', action='store_true', help='print log')
    parser.add_argument('-b', '--batch_size', required=True, type=int, help='batch size : 20, 10')
    opt = parser.parse_args()

    file = open(opt.infile)

    batch_size = opt.batch_size
    correct_sentence_slots, correct_sentence_intents, correct_sentence, sentence_number = 0.0, 0.0, 0.0, 0.0
    line_idx = 0
    fscores = []
    all_TP, all_FP, all_FN, all_TN = 0.0, 0.0, 0.0, 0.0
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    TP2, FP2, FN2, TN2 = 0.0, 0.0, 0.0, 0.0
    for line in file:
        
        line = line.strip('\n\r')
        if ' : ' in line:
            line_num, line = line.split(' : ')
        tmps = line.split(' <=> ')
        if len(tmps) > 1:
            line, intent_label, intent_pred = tmps
            intent_label_items = set(intent_label.split(';')) if intent_label != '' else set()
            intent_pred_items = set(intent_pred.split(';')) if intent_pred != '' else set()
            for pred_intent in intent_pred_items:
                if pred_intent in intent_label_items:
                    TP2 += 1
                else:
                    FP2 += 1
            for label_intent in intent_label_items:
                if label_intent not in intent_pred_items:
                    FN2 += 1
            correct_sentence_intents += int(intent_label_items == intent_pred_items)
            intent_correct = (intent_label_items == intent_pred_items)
        else:
            line = tmps[0]
            intent_correct = True
        sentence_number += 1

        words, labels, preds = [], [], []
        items = line.split(' ')
        for item in items:
            parts = item.split(':')
            word, pred, label = ':'.join(parts[:-2]), parts[-2], parts[-1]
            words.append(word)
            labels.append(label)
            preds.append(pred)
        label_chunks = set(get_chunks(['O']+labels+['O']))
        pred_chunks = set(get_chunks(['O']+preds+['O']))
        for pred_chunk in pred_chunks:
            if pred_chunk in label_chunks:
                TP += 1
            else:
                FP += 1
        for label_chunk in label_chunks:
            if label_chunk not in pred_chunks:
                FN += 1
        correct_sentence_slots += int(label_chunks == pred_chunks)
        if intent_correct and label_chunks == pred_chunks:
            correct_sentence += 1
        if label_chunks != pred_chunks and opt.print_log:
            print(' '.join([word if label == 'O' else word+':'+label for word, label in zip(words, labels)]))
            print(' '.join([word if pred == 'O' else word+':'+pred for word, pred in zip(words, preds)]))
            print('-'*20)

        line_idx += 1
        if line_idx == batch_size:
            all_TP += TP
            all_FP += FP
            all_FN += FN
            f = 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN != 0 else 0
            fscores.append(f)
            line_idx = 0
            TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
            TP2, FP2, FN2, TN2 = 0.0, 0.0, 0.0, 0.0
    
    print(all_TP, all_FP, all_FN, 2 * all_TP / (2 * all_TP + all_FP + all_FN))
    print(sum(fscores) / len(fscores))
