import torch
import argparse
# need the line below if running Python interpreter
# from onmt.inputters.text_dataset import TextMultiField

# just in case you want to look at freqs or some other dict
def dump(d, fname):
    import json
    with open(fname, 'w') as f:
        json.dump(vocab, f, indent=4)

def get_words(vocab_path):
    d = torch.load(vocab_path)
    _, f = d['src'].fields[0]
    vocab = vars(f.vocab)
    words = vocab['freqs'].keys()
    return words

def oov(ref_words, comp_words):
    ref_set, comp_set = set(ref_words), set(comp_words)

    # ref_words and comp_words should have no duplicates
    # (since they come from dict keys)
    assert(len(ref_set) == len(ref_words))
    assert(len(comp_set) == len(comp_words))

    oov_set = comp_set - ref_set

    # % of words in comp_words that don't occur in ref_words
    return len(oov_set) / float(len(comp_set))

def main(args):
    ref_words = get_words(args['ref_vocab'])
    print('Ref: {}'.format(args['ref_vocab']))
    print('* Size of ref vocab: {}'.format(len(ref_words)))

    comp_words = get_words(args['comp_vocab'])
    print('Comp: {}'.format(args['comp_vocab']))
    print('* Size of comp vocab: {}'.format(len(comp_words)))

    oov_rate = oov(ref_words, comp_words)
    print('OOV rate: {}'.format(oov_rate))

def _get_parser():
    parser = argparse.ArgumentParser(description='oov.py')
    parser.add_argument('--ref_vocab', '-ref_vocab', type=str,
                        help='path to .vocab.pt file to load')
    parser.add_argument('--comp_vocab', '-comp_vocab', type=str,
                        help='path to .vocab.pt file to load')
    return parser

if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    main(vars(args))