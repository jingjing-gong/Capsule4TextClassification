import operator
import _pickle as pkl
from collections import defaultdict

class Vocab(object):
    def __init__(self, id_start=5):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.id_start = id_start

    def add_word(self, word, count=1):
        word = word.strip()
        if len(word) == 0:
            return
        elif word.isspace():
            return
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def construct(self, words):
        for word in words:
            self.add_word(word)
        total_words = float(sum(self.word_freq.values()))

        '''sort by word frequency'''
        new_word_to_index = {}
        new_index_to_word = {}
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        self.word_freq = dict(sorted_tup)
        for idx, (word, freq) in enumerate(sorted_tup):
            index = self.id_start + idx
            new_word_to_index[word] = index
            new_index_to_word[index] = word

        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word

        print('{} total words with {} uniques'.format(total_words, len(self.word_freq)))

    def limit_vocab_length(self, base_freq):
        """
        Truncate vocabulary to keep most frequent words

        Args:
            None

        Returns:
            None
        """

        new_word_to_index = {}
        new_index_to_word = {}
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        vocab_tup = [item for item in sorted_tup if item[1] > base_freq]
        self.word_freq = dict(vocab_tup)
        for idx, (word, freq) in enumerate(vocab_tup):
            index = self.id_start + idx
            new_word_to_index[word] = index
            new_index_to_word[index] = word
        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word

    def save_vocab(self, filePath):
        """
        Save vocabulary a offline file

        Args:
            filePath: where you want to save your vocabulary, every line in the
            file represents a word with a tab seperating word and it's frequency

        Returns:
            None
        """
        with open(filePath, 'wb') as fd:
            pkl.dump(self.word_to_index, fd)
            pkl.dump(self.index_to_word, fd)
            pkl.dump(self.word_freq, fd)

    def load_vocab_from_file(self, filePath):
        """
        Truncate vocabulary to keep most frequent words

        Args:
            filePath: vocabulary file path, every line in the file represents
                a word with a tab seperating word and it's frequency

        Returns:
            None
        """
        with open(filePath, 'rb') as fd:
            self.word_to_index = pkl.load(fd)
            self.index_to_word = pkl.load(fd)
            self.word_freq = pkl.load(fd)

        print('load from <' + filePath + '>, there are {} words in dictionary'.format(len(self.word_freq)))

    def encode(self, word):
        if word not in self.word_to_index:
            return 1    #unk
        else:
            return self.word_to_index[word]

    def decode(self, index):
        if index not in self.index_to_word:
            return 'pad/unk'
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_to_index)