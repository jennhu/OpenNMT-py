import torch
import argparse
# need the line below if running Python interpreter
# from onmt.inputters.text_dataset import TextMultiField

def _dump(d, fname):
    import json
    with open(fname, 'w') as f:
        json.dump(d, f, indent=4)

def _write_list(l, fname):
    with open(fname, 'w') as f:
        for item in l:
            f.write(str(item) + '\n')

def _flatten(l):
    return [item for sublist in l for item in sublist]

def _get_vocab(vocab_path):
    d = torch.load(vocab_path)
    _, f = d['src'].fields[0]
    return f.vocab

def _get_word_list(vocab):
    return vocab.itos

def _get_freq_counter(vocab):
    return vocab.freqs

def count_k_freq(ref_vocab, comp_vocab, k):
    '''
    Counts how many elements of comp_words appear k times in ref_vocab
    '''
    freqs = _get_freq_counter(ref_vocab)
    comp_words = _get_word_list(comp_vocab)
    words = []
    for w in comp_words:
        if w in ['<unk>', '<blank>']:
            continue
        else:
            try:
                if freqs[w] == k:
                    words.append(w)
            except KeyError:
                # this shouldn't happen if we use combined vocab
                if k == 0:
                    words.append(w)
                continue
    return words

def oov(ref_words, comp_words):
    # ref_words and comp_words should have no duplicates
    ref_set, comp_set = set(ref_words), set(comp_words)
    assert(len(ref_set) == len(ref_words))
    assert(len(comp_set) == len(comp_words))

    # % of words in comp_words that don't occur in ref_words
    oov_set = comp_set - ref_set
    return len(oov_set) / float(len(comp_set))

def combine_vocab(ref_vocab, comp_vocab):
    ref_words = _get_word_list(ref_vocab)
    comp_words = _get_word_list(comp_vocab)
    combined = list(set(ref_words + comp_words))
    assert(oov(combined, comp_words) == 0)
    return combined

def main(args):
    ref_vocab = _get_vocab(args['ref'])
    comp_vocab = _get_vocab(args['comp'])

    if args['mode'] == 'oov':
        ref_words = _get_word_list(ref_vocab)
        print('Ref: {}'.format(args['ref']))
        print('* Size of ref vocab: {}'.format(len(ref_words)))

        comp_words = _get_word_list(comp_vocab)
        print('Comp: {}'.format(args['comp']))
        print('* Size of comp vocab: {}'.format(len(comp_words)))

        oov_rate = oov(ref_words, comp_words)
        print('OOV rate: {}'.format(oov_rate))

    elif args['mode'] == 'combine':
        combined = combine_vocab(ref_vocab, comp_vocab, args['combine'])
        _write_list(combined, args['save'])

    else:
        freqs = {
            k : count_k_freq(ref_vocab, comp_vocab, k) for k in range(11)
        }
        _dump(freqs, args['save'])

        # count how many sentences have at least one of the infrequent words
        # with open('data/iarpa/iarpa.tokenized', 'r') as f:
        #     sents = f.readlines()
        # sents = [s.strip('\n') for s in sents]
        # infreq_words = _flatten(freqs.values())
        # bad_sents = [s for s in sents if any(w in s for w in infreq_words)]
        # print(len(bad_sents))

def _get_parser():
    parser = argparse.ArgumentParser(description='oov.py')
    parser.add_argument('--ref', '-ref', type=str,
                        help='path to reference .vocab.pt file')
    parser.add_argument('--comp', '-comp', type=str,
                        help='path to comparison .vocab.pt file')
    parser.add_argument('--mode', '-mode', choices=['oov', 'combine', 'freq'],
                        help='(oov) calculate oov rate of comp w.r.t. ref; '
                             '(combine) combine the vocab files; '
                             '(freq) count num comp words that occur k times '
                             'in ref vocab')
    parser.add_argument('--save', '-save', type=str,
                        help='path to save output file (depends on mode)')
    return parser

if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    main(vars(args))
