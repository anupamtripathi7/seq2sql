import json
import unicodedata
from tqdm import tqdm
from utils import load_pickle, save_pickle
from itertools import chain
from copy import deepcopy
questions_path = 'data/questions/'
sql_queries_path = 'data/sql_queries/'
word_idx_mappings_path = 'data/word_idx_mappings/'
wiki_sql_path = 'data/WikiSQL_files/'
# coding=utf8


def read_json(filename):
    """
    Args:
        filename (str): name of the json file
    Returns:
        data (list of dictionaries): contains the questions, labels
    """
    with open(filename, encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def replace_all(text, dict_):
    """
    Replaces all instances of dictionary keys with dictionary values in text
    Args:
        text (str): string in which particular words are to be replaced
        dict_ (dict): key string is replaced with value string

    Returns:
        text (str): string after replacement
    """
    for i, j in dict_.items():
        text = text.replace(i, j)
    return text


def get_questions(json_data, mode):
    """
    Args:
        json_data (list of dictionaries): contains the questions, labels
        mode : train/test

    Returns:
        questions: features
    """
    questions = []
    to_replace = {"'s": " 's", "?": " ?", ",": " ,", '"': ' " ', "\t": ''}
    if mode == 'train':
        for idx, item in enumerate(tqdm(json_data)):
            question = replace_all(item['question'], to_replace)
            question = question.strip()
            questions.append('<s> ' + question + ' </s>')

    if mode == 'test':
        for idx, item in enumerate(tqdm(json_data)):
            question = replace_all(item['question'], to_replace)
            question = " "+question+" "
            for word in question.split(' '):
                if word not in vocab and word != '':
                    question = question.replace(" "+word+" ", ' <UNK> ')
            question = question.strip()
            questions.append('<s> ' + question + ' </s>')

    return questions


def get_sql(json_data, tables, mode='train'):
    """
    Returns the sql query using sql json and database format
    Args:
        json_data (list of dictionaries): contains the questions, labels
        tables (list of dictionaries): contains the database schema
        mode: train/test

    Returns:
        sql: label(sql query)

    """
    sql = []

    for idx, item in enumerate(tqdm(json_data)):
        for idx2, table_item in enumerate(tables):
            if table_item['id'] == item['table_id']:
                if len(item['sql']['conds']) > 1:
                    where = []
                    where.extend([table_item['header'][int(item['sql']['conds'][0][0])],
                                  Query.cond_ops[int(item['sql']['conds'][0][1])], str(item['sql']['conds'][0][2])])
                    for i in range(1, len(item['sql']['conds'])):
                        where.extend(['AND', table_item['header'][int(item['sql']['conds'][i][0])],
                                      Query.cond_ops[int(item['sql']['conds'][i][1])], str(item['sql']['conds'][i][2])])
                    where_string = ' '.join(map(str, where))
                    sql.append("<s> SELECT " + Query.agg_ops[int(item['sql']['agg'])] + " " + table_item['header'][
                        int(item['sql']['sel'])] + ' FROM ' + table_item['id'] + ' WHERE ' + where_string + ' </s>')

                elif not item['sql']['conds']:
                    sql.append("<s> SELECT " + Query.agg_ops[int(item['sql']['agg'])] + " " +
                               table_item['header'][int(item['sql']['sel'])] + ' FROM ' + table_item['id'] + ' </s>')

                else:
                    sql.append("<s> SELECT " + Query.agg_ops[int(item['sql']['agg'])] + " " + table_item['header'][
                        int(item['sql']['sel'])] + ' FROM ' + table_item['id'] + ' WHERE ' + table_item['header'][
                                   int(item['sql']['conds'][0][0])]
                               + " " + Query.cond_ops[int(item['sql']['conds'][0][1])] + " " + str(
                        item['sql']['conds'][0][2]) + ' </s>')

    if mode == 'test':
        for i, line in enumerate(sql):
            for j, word in enumerate(sql[i].split(' ')):
                if word not in vocab and word != '':
                    sql[i] = ' ' + sql[i] + ' '
                    sql[i] = sql[i].replace(" " + word + " ", ' <UNK> ')
                    sql[i] = sql[i].strip()

    return sql


def construct_vocab(lines, sql, traintables, testtables):
    """

    Args:
        lines (list of string): questions/features
        sql (list of strings): sql queries/labels
        traintables (list of dictionaries): database tables for train data
        testtables (list of dictionaries): database tables for test data

    Returns:
        vocabulary (list of strings)

    """
    for train_table in traintables:
        lines.append(str(train_table['id']))
        lines.extend([str(item) for item in train_table['header']])
    for test_table in testtables:
        lines.append(str(test_table['id']))
        lines.extend([str(item) for item in test_table['header']])

    lines.extend(sql)
    vocabulary = set(chain(*(line.split() for line in lines if line)))
    return vocabulary


def construct_word2vec(vocabulary):
    """
    Args:
        vocabulary: list of vocabularies

    Returns: Two dictionaries of mappings between words and indices
    """

    word_to_idx_dict = {item: ii for ii, item in enumerate(vocabulary)}
    word_to_idx_dict['<UNK>'] = len(word_to_idx_dict.keys())
    idx_to_word_dict = {ii: word for word, ii in word_to_idx_dict.items()}
    return word_to_idx_dict, idx_to_word_dict


def tokenize(data):
    """
    Args:
        data (list of strings): data to be tokenized
    Returns:
        tokenized string data (list of lists)
    """
    tokenized = []
    for idx, item in enumerate(data):
        tokenized.append(item.split(' '))
    return tokenized


def questions_to_vector(data, word_2_idx):
    """
    Args:
        data (list of list): tokenized string data
        word_2_idx (dict): word to index mappings
    Returns:
        tokenized vector data
    """
    for idx1, item in enumerate(data):
        while '' in item:
            item.remove('')
        for idx2, word in enumerate(item):
            data[idx1][idx2] = word_2_idx[word]
    return data


def vector_to_question(data, idx_2_word):
    """
    Args:
        data (list of list): tokenized vector data
        idx_2_word (dict): index to word mappings
    Returns:
        tokenized string data
    """
    for idx1, item in enumerate(data):
        for idx2, word in enumerate(item):
            data[idx1][idx2] = idx_2_word[word]
    return data


def unicode2ascii(data):
    """
    Args:
        data (list of strings): data in unicode format
    Returns:
        data in ascii format
    """
    for i, item in enumerate(data):
        data[i] = unicodedata.normalize("NFKD", item)
    return data


def unicode2ascii_tables(tables):
    """
    Args:
        tables (list of dictionaries): unicode data
    Returns:
        tables in ascii format
    """
    for i, table in enumerate(tables):
        tables[i]['id'] = unicodedata.normalize("NFKD", table['id'])
        for j, head in enumerate(table['header']):
            tables[i]['header'][j] = unicodedata.normalize("NFKD", head)
    return tables


if __name__ == "__main__":
    # read json files in a list of dictionaries
    train = read_json(wiki_sql_path + "train.jsonl")
    test = read_json(wiki_sql_path + "test.jsonl")
    train_tables = read_json(wiki_sql_path + "train.tables.jsonl")
    test_tables = read_json(wiki_sql_path + "test.tables.jsonl")

    train_questions = get_questions(train, mode='train')
    train_sql = get_sql(train, train_tables, mode='train')

    train_questions = unicode2ascii(train_questions)
    train_sql = unicode2ascii(train_sql)
    train_tables = unicode2ascii_tables(train_tables)
    test_tables = unicode2ascii_tables(test_tables)

    train_questions_copy = deepcopy(train_questions)
    vocab = construct_vocab(train_questions_copy, train_sql, train_tables, test_tables)
    word2idx, idx2word = construct_word2vec(vocab)

    test_questions = get_questions(test, 'test')
    test_sql = get_sql(test, test_tables, 'test')

    test_questions = unicode2ascii(test_questions)
    test_sql = unicode2ascii(test_sql)

    train_questions_tokenized = tokenize(train_questions)
    test_questions_tokenized = tokenize(test_questions)
    train_sql_tokenized = tokenize(train_sql)
    test_sql_tokenized = tokenize(test_sql)

    train_questions_tokenized = questions_to_vector(train_questions_tokenized, word2idx)
    train_sql_tokenized = questions_to_vector(train_sql_tokenized, word2idx)
    test_questions_tokenized = questions_to_vector(test_questions_tokenized, word2idx)
    test_sql_tokenized = questions_to_vector(test_sql_tokenized, word2idx)

    save_pickle(train_questions_tokenized, questions_path + 'train_questions_tokenized.pkl')
    save_pickle(test_questions_tokenized, questions_path + 'test_questions_tokenized.pkl')
    save_pickle(train_sql_tokenized, sql_queries_path + 'train_sql_tokenized.pkl')
    save_pickle(test_sql_tokenized, sql_queries_path + 'test_sql_tokenized.pkl')