import collections
import json
import math
import re
import datetime
import traceback

class TextCleaner:
    def __init__(self):
        self.updated_time = datetime.datetime(1971,1,1)
        self.model = {}
        self.identifier = None
        pass

    ##====== METHOD NEED TO BE IMPLEMENTED BY CHILD CLASS ============
    # Build cleaner model (if no need, just create a function with pass)
    def build_model(self, documents):
        raise Exception('[TextCleaner] Method build_model is not implemented in child class!')
    # Generate clean word list
    def clean_word_list(self, t):
        raise Exception('[TextCleaner] Method clean_word_list is not implemented in child class!')
    # Generate bag of words
    def bow(self, t):
        raise Exception('[TextCleaner] Method bow is not implemented in child class!')
    # Clean the input text
    def clean_text(self, t):
        raise Exception('[TextCleaner] Method clean_text is not implemented in child class!')

    ##====== METHOD SHARED BY ALL CHILD CLASS ===============
    # Load from external model
    def load(self, model):
        self.model = model
    # Get the method identifier
    def get_identifier(self):
        if (self.identifier == None):
            raise Exception('[TextCleaner] Variable identifier is not implemented in child class!')
        return self.identifier

########## Initial method classes ###########
# The initial method text cleaner
class InitialMethodTextCleaner(TextCleaner):
    def __init__(self):
        TextCleaner.__init__(self)
        self.identifier = 'InitialMethodTextCleaner'

        self.opposite_words = set(['kurang','ga', 'g', 'gg', 'tidak', 'tak', 'tdk', 'tdak', 'enggak', 'engga', 'kagak', 'gak', 'ngak', 'nga', 'ngk', 'nggk', 'gk', 'gx', 'ngga', 'gax', 'gag', 'nda', 'ndak'])
        self.used_opp_word = 'ga'
        self.stop_words = set(['yang', 'di', 'ke', 'untuk', 'dan', 'and', 'lah', 'eh', 'juga', 'pun', 'the', 'dengan', 'an', 'loh', 'dn', 'nya', 'pada', 'tu', 'mah', 'kan', 'hm',
                    'aku', 'kamu', 'you'])
        self.model['synonym_words'] = {}

        # Exceptions set:
        self.nya_exception = set(['tanya', 'ditanya', 'punya'])
        self.ku_exception = set(['anaku', 'thanku', 'buku'])
        self.x_exception = set(['diinbox', 'inbox', 'thx', 'thnx', 'vermax'])
        self.di_exception = set(['diri', 'disc', 'digit', 'diet', 'diskon', 'discount', 'dikit' ])
        self.quote_exception = set(['jum\'at'])

    def build_model(self, documents):
        pass

    def clean_word_list(self, t):
        word_list = self._preprocess_text(t).split()
        word_list = [self._preprocess_word(w) for w in word_list if self._filter_word_1(w)]
        word_list = [self._unified_to_synonym(w) for w in word_list if self._filter_word_2(w)]
        word_list = [w for w in word_list if self._filter_word_3(w)]
        word_list = self._combine_opposite_word_with_next_word(word_list)
        return word_list

    def bow(self, t):
        word_list = self.clean_word_list(t)
        bow_result = collections.Counter()
        for i, word in enumerate(word_list):
            bow_result[word] += 1
            # Use bigram
            if (i < len(word_list)-1):
                next_word = word_list[i+1]
                bigram = word + '_' + next_word
                bow_result[bigram] += 1
                # And trigram
                if (i < len(word_list)-2):
                    next_next_word = word_list[i+2]
                    trigram = bigram + '_' + next_next_word
                    bow_result[trigram] += 1
        return bow_result

    def clean_text(self, t):
        word_list = self.clean_word_list(t)
        return ' '.join(word_list)

    def _filter_stop_word(self, w):
        return not w in self.stop_words

    def _filter_short_word(self, w):
        return len(w) > 1

    def _filter_word_1(self, w):
        return self._filter_stop_word(w)

    def _filter_word_2(self, w):
        return self._filter_short_word(w)

    def _filter_word_3(self, w):
        return self._filter_stop_word(w)

    def _remove_symbols(self, t):
        t = re.sub('[^a-zA-Z-\s\']+', ' ', t)
        t = re.sub('[\']+', '', t)
        return t

    def _remove_consecutive_letters(self, t):
        return re.sub(r'([uthk])\1+', r'\1', t)

    def _remove_3_consecutive_letters(self, t):
        return re.sub(r'([a-z])\1\1+', r'\1', t)

    def _preprocess_text(self,t):
        t = t.lower()
        #rev removing symbol
        #t = self._remove_symbols(t)
        t = self._remove_3_consecutive_letters(t)
        t = self._remove_consecutive_letters(t)
        return t

    def _remove_nya(self, w):
        if w in self.nya_exception:
            return w
        elif (len(w) > 4 or len(w) == 3) and w[-3:] == 'nya':
            return w[:-3]
        elif (len(w) > 3 or len(w) == 2) and w[-2:] == 'ny':
            return w[:-2]
        return w

    def _remove_ku(self,w):
        if (not w in self.ku_exception) and (len(w) > 4 or len(w) == 2) and w[-2:] == 'ku':
            return w[:-2]
        return w

    def _remove_x(self, w):
        if (not w in self.x_exception) and (len(w) > 3 or len(w) == 1) and w[-1:] == 'x':
            return w[:-1]
        return w

    def _remove_di(self, w):
        if (not w in self.di_exception) and (len(w) > 4 or len(w) == 2) and w[:2] == 'di':
            return w[2:]
        return w

    def _remove_quote(self, w):
        if w in self.quote_exception:
            return re.sub(r'\'', r'', w)
        return re.sub(r'\'', ' ', w)

    def _unified_opposite_words(self, w):
        if w in self.opposite_words:
            return self.used_opp_word
        else:
            return w

    def _remove_end_in_consecutive_letter(self, w):
        return re.sub(r'([a-km-oq-rt-z])\1+$', r'\1', w)

    def _preprocess_word(self, w):
        w = self._remove_end_in_consecutive_letter(w)
        w = self._remove_quote(w)
        w = self._remove_x(w)
        w = self._remove_ku(w)
        w = self._remove_nya(w)
        w = self._remove_di(w)
        w = self._unified_opposite_words(w)
        w = self._remove_end_in_consecutive_letter(w)
        return w

    def _unified_to_synonym(self, w):
        result = self.model['synonym_words'].get(w)
        if not result == None:
            return result
        return w

    def _combine_opposite_word_with_next_word(self, ws):
        opp_word_location = []
        for i,w in enumerate(ws):
            if w == self.used_opp_word:
                opp_word_location.append(i)

        for i in reversed(opp_word_location):
            if i+1 >= len(ws):
                continue
            new_word = ws[i] + '_' +  ws[i+1]
            ws[i] = new_word
            del ws[i+1]
        return ws

