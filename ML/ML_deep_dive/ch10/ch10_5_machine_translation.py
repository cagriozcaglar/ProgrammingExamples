import os
import torch
from d2l import torch as d2l

# 10.5.1. Download and preprocess dataset
class MTFraEng(d2l.DataModule):  #@save
    """The English-French dataset."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root,
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()

# Download dataset
data = MTFraEng()
raw_text = data._download()
print(raw_text[:75])
len(raw_text)
# 11489286

# Preprocess dataset
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())] 
    return ''.join(out)

text = data._preprocess(raw_text)
print(text[:80])

# 10.5.2. Tokenization
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text: str, max_examples = None):
    source, target = [], []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if max_examples and i > max_examples:
            break
        # Split line
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            source.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            target.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return source, target

source, target = data._tokenize(text)
source[:6], target[:6]
'''
([['go', '.', '<eos>'],
  ['hi', '.', '<eos>'],
  ['run', '!', '<eos>'],
  ['run', '!', '<eos>'],
  ['who', '?', '<eos>'],
  ['wow', '!', '<eos>']],
 [['va', '!', '<eos>'],
  ['salut', '!', '<eos>'],
  ['cours', '!', '<eos>'],
  ['courez', '!', '<eos>'],
  ['qui', '?', '<eos>'],
  ['ça', 'alors', '!', '<eos>']])
'''

#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence','count', source, target)

# Loading sequences of fixed length
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.source_vocab, self.target_vocab = self._build_arrays(self._download())

@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, source_vocab:None, target_vocab=None):
    def _build_array(sentences, vocab, is_target=False):
        # Pad sentences to the same length
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq))
        )
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        # If sentence is a target, prepend <bos>
        if is_target:
            sentences = [ ['<bos>'] + s for s in sentences]
        # If vocab is None, create vocab from sentence,
        # with min_freq=2 (anything below is converted to <unk>)
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        # Create vocabulary tensor as an array
        array = torch.tensor([vocab[s] for s in sentences])
        valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
        return array, vocab, valid_len

# 10.5.4. Reading the dataset
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)

data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', src.type(torch.int32))
print('decoder input:', tgt.type(torch.int32))
print('source len excluding pad:', src_valid_len.type(torch.int32))
print('label:', label.type(torch.int32))