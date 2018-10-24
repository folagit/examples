'''

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.840B.300d.zip
(source page: http://nlp.stanford.edu/projects/glove/)
'''

import pandas as pd, numpy as np, os, errno
from sklearn.model_selection import StratifiedShuffleSplit as sssplit
# from gensim.corpora import dictionary as gdict
from collections import defaultdict
import re
import nltk
import itertools
import gensim
import os
import urllib
import zipfile
import requests
from io import BytesIO
import pickle
from utils.parser import local_args
from utils.spellcheck import sym_spell


args = local_args()
np.random.seed(args.seed)

def init():

    global glove_dir, glove_file, glove_pkl, glove_url
    global data_dir, did_dir
    global wv_dim, vocabulary_size, misc_symbols
    global unknown_num_token, unknown_word_token, new_line_token
    global charmap

    glove_dir = os.path.join(os.environ['HOME'], 'data', 'glove')
    glove_file = 'glove.840B.300d.txt'
    glove_pkl = 'glove_embeddings.pkl'
    glove_url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    data_dir = 'data'
    did_dir =  os.path.join(os.environ['HOME'],'data','deidentified')

    wv_dim = 300
    vocabulary_size = 100000
    misc_symbols = {'(': 'open_parentheses_symbol',
                    ')': 'close_parentheses_symbol',
                    '[': 'open_square_symbol' ,
                    ']': 'close_square_symbol',
                    '{': 'open_brace_symbol',
                    '}': 'close_brace_symbol',
                    '<': 'open_angle_symbol',
                    '>': 'close_angle_symbol',
                    '-': 'hyphen_symbol',
                    ';': 'semi_colon_symbol',
                    '.': 'period_symbol',
                    '/': 'back_slash_symbol',
                    '\\': 'forward_slash_symbol',
                    '*': 'asterisk_symbol_symbol',
                    '|': 'vertical_bar_symbol',
                    ',': 'comma_symbol',
                    '?': 'question_mark_symbol',
                    '!': 'exclamation_mark_symbol',
                    ':': 'colon_symbol',
                    '_': 'under_score_symbol',
                    '\"': 'double_quotation_mark_symbol',
                    '’': 'back_apostrophe_symbol',
                    '\'': 'single_quotation_mark_symbol',
                    '‘': 'forward_apostrophe_symbol',
                    '@': 'at_sign_symbol',
                    '#': 'hash_tag_symbol',
                    '$': 'dollar_sign_symbol',
                    '%': 'percentage_sign_symbol',
                    '^': 'caret_sign_symbol',
                    '&': 'ampersand_symbol',
                    '~': 'tilde_symbol',
                    '+': 'plus_sign_symbol',
                    '-': 'minus_sign_symbol',
                    '=': 'equal_sign_symbol'}


    unknown_word_token = 'UNKNOWN_WORD_TOKEN'.lower()
    new_line_token = 'NEW_LINE_TOKEN'.lower()
    unknown_num_token = 'UNKNOWN_NUMERIC_TOKEN'.lower()

    cmap = list('abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\"’\'/\|_@#$%ˆ&*˜‘+-=<>()[]{} ')
    cmap.append('new_line_token')
    charmap = pd.Series(data=np.arange(len(cmap)), index=cmap)


def save_pkl(fdir=None, f=None, obj=None):
    if fdir is None:
        fdir = ''
    path = os.path.join(args.tag,fdir)
    saveas = os.path.join(path, str(f) + '.pkl')
    #print(saveas, '... saving ...')
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            raise OSError('can\'t create directory %s' % path)
    elif os.path.isfile(saveas):
        try:
            os.remove(saveas)
        except:
            raise OSError('can\'t remove file %s' % saveas)

    with open(saveas, mode='wb') as output:
        pickler = pickle.Pickler(output, -1)
        pickler.dump(obj)
        output.close()


def load_pkl(fdir=None, f=None):
    if fdir is None:
        fdir = ''
    # saveas = dir + str(f) + '.pkl'
    path = os.path.join(args.tag,fdir)
    loadas = os.path.join(path, str(f) + '.pkl')
    #print(loadas,'... loading ....')

    if os.path.isfile(loadas):
        try:
            input = open(loadas, mode='rb')
            dat = pickle.load(input)
            input.close()
            return dat
        except:
            raise OSError('can\'t open file %s' % loadas)

    return None



def _create_report_dataframe():
    data = None
    global did_dir
    try:
        data = pd.read_csv(filepath_or_buffer=os.path.join(did_dir,'fileList.cv'), header=None, names=['filename', 'label'],
                           delimiter=',', index_col='filename')
        report_path = os.path.join(did_dir,'reports')
        reports = []
        for filename in data.index:
            with open(file=os.path.join(report_path,filename), mode='r') as f:
                lines = np.array([line.rstrip() for line in f.read().splitlines()])
                reports.append(lines[~(lines == '')])
        data['report'] = pd.Series(data=reports, index=data.index)

    except FileNotFoundError:
        print('Wrong file name or file path!')
        return data

    save_pkl(fdir=did_dir, f='report', obj=data)
    return data


def get_reports_dataframe():
    global did_dir
    try:
        data = pd.read_pickle(path=os.path.join(did_dir,'reports.pkl'))
        return data
    except FileNotFoundError:
        print('Creating report dataframe...')
        return _create_report_dataframe()


def _preprocess_data():
    '''Tokenize text into character encoding or word token encoding

    :return:
        Tuple of tuples: numpy arrays: (X (token encoded reports), y (report labels)), and a word2index containing
        words and their frequencies in the corpus
    '''
    global did_dir,sym_spell
    X, y = (None,None)


    X,y = load_pkl(fdir=did_dir, f='Xy.pkl')
    vocabulary = load_pkl(fdir=did_dir,f='vocabulary.pkl')

    if X is None:
        print('Preprocessing data...')
        min_substring = 2
        data = get_reports_dataframe()
        X, y, rid = data['report'].values, data['label'].values, data.index.values
        X = [[token for line in report for token in line.lower().split() if len(token) <= 26] for report in _preprocess_text(X)]
        xdict, xfreq = list(zip(*nltk.FreqDist(itertools.chain(*X)).most_common()))
        vocabulary = pd.Series(data=xfreq, index=list(xdict))
        X = np.asarray(X)
        y = np.asarray(y)

        if vocabulary is None:
            print('Preprocessing vocabulary...')

            xdict, xfreq = list(zip(*nltk.FreqDist(itertools.chain(*X)).most_common()))
            vocabulary = pd.Series(data=np.arange(len(list(xdict))), index=list(xdict))


        gembs = get_glove_embeddings()
        lookup = gembs.index.values
        lookup = ([word for word in lookup if len(word) >= min_substring])
        lookup = pd.Series(data=np.ones((len(lookup),)), index=lookup)

        Xnew = []
        for x in X:
            doc = []
            for xi in x:
                # if word not in lookup:
                if word not in lookup:
                    suggs = sym_spell.lookup_compound(word,1)
                if type(word) is list:
                    line.extend(word)
                else:
                    line.append(word)
                # else:
                # line.append(word)
            Xnew.append(line)
        print('word correction complete')
        # X0 = [[wc.viterbi_segment(word) if word not in lookup else word for word in line] for line in X]
        X = np.array(Xnew)
        save_pkl(fdir=data_dir,f='Xy',obj=(X,y))
        save_pkl(fdir=did_dir, f='vocabulary', obj=vocabulary)



    return (X, y), vocabulary


def get_vocabulary():
    vocabulary = load_pkl(fdir=did_dir,f='vocabulary')
    if vocabulary:
        return vocabulary
    print('preprocessing word2index ....')
    _, vocabulary = _preprocess_data()
    return vocabulary


def get_index2word():
    '''
    Word lookup is a mapping of indices to tokens and vice versa.
    Each entity has a double entry allowing user to pull the token by passing an index or pull index by passing a token string
    :return: index2word series containing index/token and token/index mappings.
    '''
    global did_dir

    try:
        index2word = pd.read_pickle(path=os.path.join(did_dir,'index2word.pkl'))
        return index2word
    except FileNotFoundError:
        print('index2word has not been generated....')


def _preprocess_text(texts):
    '''
    Preprocesses deidentified text
    :param texts: list of deidentified pathology reports (each report is an array of lines within the report)
    :return: list
    '''
    global misc_symbols

    def filter_periods(s,symbol='.'):
        words = s.split(' ')
        newS = []
        for word in words:
            lenword = len(word)
            loc = [pos for pos, char in enumerate(word) if char == symbol]
            if loc:
                if loc[0] == 0:
                    word = ' ' + symbol + ' ' + word[1:]
                if loc[-1] == lenword - 1:
                    word = word[:-1] + ' ' + symbol + ' '
            newS.append(word)
        s = ' '.join(newS)
        return s

    def edit_special_characters(s):
        s = s.replace('\\X0D\\\\X0A\\', ' ')  # remove unwanted encoding string
        s = s.replace('\n', ' ')  # remove line feed (DONE BY RAKE)
        s = s.replace('\r', ' ')  # remove carriage return
        # s = s.replace('\'', '')
        s = s.replace('o clock', 'oclock')
        # s = s.replace('_', '')
        # s = re.sub(r'<.*?>', ' ', s)                                      #remove all <...> tags
        # s = re.sub(r'\*+\w*(\[[\w\s\.]*\])?\S*(?=\s)', ' ', s)            #remove all **[...] tags
        # s = re.sub(r'\b[1-9]:00\b|\b0[1-9]:00\b|\b1[0-2]:00\b', repl, s)  #replace all timestamps
        s = unify_clocks(s)
        # s = s.replace('\'', ' ')
        # s = re.sub('[^a-zA-Z.]', ' ', s)
        s = re.sub(r'(p\.?m\.?)', ' pm ', s, flags=re.IGNORECASE)
        s = re.sub(r'(a\.?m\.?)', ' am ', s, flags=re.IGNORECASE)
        s = re.sub(r'(dr\.)', 'dr', s, flags=re.IGNORECASE)
        re.sub(r'\.{1}', ' ', s)
        for symbol, token in misc_symbols.items():
            if symbol in '.:' and symbol in s:
                s = filter_periods(s,symbol=symbol)
            else:
                s = s.replace(symbol, ' '+symbol+' ')
        re.sub(r'\.{2,}', ' ', s)
        return ' '.join(s.lower().split())

    def unify_clocks(s):
        nums = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
                10: 'ten', 11: 'eleven', 12: 'twelve'}
        for i in range(1, 13):
            s = re.sub(r'(0?' + str(i) + ':00 *?oclock)', str(i) + ' oclock', s,
                       flags=re.IGNORECASE)  # (0)x:00 oclock => x oclock
            s = re.sub(r'(0?' + str(i) + ':00)', str(i) + ' oclock', s, flags=re.IGNORECASE)  # (0)x:00 => x oclock
            s = re.sub(r'( ' + str(i) + ' *?-)', ' ' + str(i) + ' oclock to ', s,
                       flags=re.IGNORECASE)  # x( )- => x oclock to
            s = re.sub(r'( ' + str(i) + ' *?oclock *?-)', ' ' + str(i) + ' oclock to ', s,
                       flags=re.IGNORECASE)  # x oclock( )- => x oclock to
            s = re.sub(r'( \"?' + str(i) + ' *?oclock)', ' ' + nums[i] + ' oclock', s,
                       flags=re.IGNORECASE)  # convert numbers to words 1 => one..

        return s

    cleaned_texts = []

    for text in texts:
        text = [edit_special_characters(line) for line in text]
        # report = ' '.join([word for word in report.split() if len(word) > 1])
        cleaned_texts.append(text)
    return cleaned_texts


def get_glove_embeddings():
    '''

    :return: a pandas series of Glove pretrained word embeddings
    '''
    global glove_dir,glove_file,glove_url,glove_pkl,wv_dim,unknown_word_token,unknown_num_token
    try:
        gembs = pd.read_pickle(path=os.path.join(glove_dir,glove_pkl))
        return gembs
    except FileNotFoundError:
        if not os.path.exists(os.path.join(glove_dir,glove_file)):
            try:
                os.makedirs(glove_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            r=requests.get(glove_url)
            with zipfile.ZipFile(BytesIO(r.content)) as zfile, open(os.path.join(glove_dir, '__init__.py'), 'w') as _:
                print('extracting glove zip file')
                zfile.extractall(glove_dir)
                print('creating __init__.py file')

        print('Preprocessing pretrained glove word embeddings...')
        gembs = {}
        with open(os.path.join(glove_dir,glove_file), encoding='utf8') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                wemb = np.asarray(values[1:], dtype='float32')
                gembs[word] = wemb

        gembs = pd.Series(gembs)
        idx = gembs.index.str.len() <= 26
        gembs = gembs[idx]
        gembs.to_pickle(path=os.path.join(glove_dir,glove_pkl))
        return gembs


def get_word_embeddings():
    global did_dir
    try:
        wembs = pd.read_pickle(path=os.path.join(did_dir,'word_embeddings.pkl'))
        return wembs
    except FileNotFoundError:
        print('processing pretrained word embeddings...')


def get_data_transformers():
    ''' This function takes a raw dictionary from corpus and maps each word to a pretrained Glove embeddings vector.
        unknown numeric and word tokens are randomly intialized.

    :return: wembs (corpus specific word embeddings, vocabulary)
    '''
    # load dictionary
    try:
        wembs = get_word_embeddings()
        index2word = get_index2word()
        return wembs, index2word
    except FileNotFoundError:
        print('Preprocessing pretrained word embeddings...')

    gembs = get_glove_embeddings()
    vocabulary = get_vocabulary()
    wembs = gembs[vocabulary.index]

    nans = wembs.isnull()  # all words in corpus word map and not in glove dictionary
    numeric = wembs.index.str.isnumeric()  # all numbers in corpus
    index = wembs.index.values
    index[np.logical_and(nans,~numeric)] = unknown_word_token  # replace non glove words with unknown token
    index[np.logical_and(nans,numeric)] = unknown_num_token  # replace non glove numbers with unknown number token
    wembs.index = index
    wembs = wembs[~wembs.index.duplicated(keep='first')]
    wembs.loc[unknown_word_token] = np.random.uniform(-.25, .25, wv_dim)  # generate random embedding
    wembs.loc[unknown_num_token] = np.random.uniform(-.25, .25, wv_dim)  # generate random embedding

    # setup new word map
    index = np.arange(len(wembs))
    index2word = pd.Series(index=wembs.index, data=index)
    wordmap = index2word.append(pd.Series(index=index, data=wembs.index))
    wembs.index = wordmap.loc[wembs.index].values

    save_pkl(fdir=did_dir, f='index2word', obj=index2word)
    save_pkl(fdir=did_dir, f='wordmap', obj=wordmap)
    save_pkl(fdir=did_dir, f='word_embeddings', obj=wembs.values)

    return wembs, index2word


def get_transformed_data():
    '''
    Function to do spell checking?

    :return:
    '''
    try:
        Xform, y = pd.read_pickle(path=os.path.join(did_dir,'transformed_Xy.pkl'))
        return Xform, y
    except FileNotFoundError:
        print('Transforming data...')
    (X, y), _ = _preprocess_data()
    wembs, index2word = get_data_transformers()
    # Xform = [[index2word[word] for word in doc] for doc in X]
    Xform = []
    for doc in X:
        docxform = []
        for word in doc:
            if word in index2word:
                docxform.append(index2word[word])
            elif WordCorrection.isnumeric(value=word):
                docxform.append(index2word.loc[unknown_num_token])
            else:
                docxform.append(index2word.loc[unknown_word_token])
        Xform.append(np.array(docxform))

    Xform = np.array(Xform)
    save_pkl(fdir=did_dir, f='transformed_Xy', obj=(Xform,y))
    return Xform,y




def load_data(n_folds=10):
    """Loads the dataset.

    # Arguments
        path: path where data resides. defaults to deidentified dataset

    # Returns
        Tuple of Numpy arrays: `(X_train, y_train), (X_test, y_test)`.
    """

    sss = sssplit(n_splits=n_folds, test_size=.1, random_state=np.random.RandomState(830452))
    X, y = get_transformed_data()
    (X_train, y_train), (X_test, y_test) = ([],[]),([],[])

    for train_idx, test_idx in sss.split(X, y):
        X_train.append(X[train_idx])
        y_train.append(y[train_idx])
        X_test.append(X[test_idx])
        y_test.append(y[test_idx])

    return (X_train, y_train), (X_test, y_test)




init()