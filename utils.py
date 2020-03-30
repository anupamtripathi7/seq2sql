import os
import torch
import pickle


def get_next_decoder(current_decoder, decoder_dict):
    """
    Switch case for getting next decoder using current decoder
    Args:
        current_decoder (str): Name of current decoder
        decoder_dict (dict): Dictionary of all decoders objects
    Returns (list): List of decoder of format [decoder(obj), decoder name(str)]

    """
    switch = {                                                            # Dictionary with next model calls
        'KeywordDecoder': [(decoder_dict['ColumnDecoder'], 'ColumnDecoder')],
        'ColumnDecoder': [(decoder_dict['OperatorDecoder'], 'OperatorDecoder')],
        'TableDecoder': [],
        'OperatorDecoder': [(decoder_dict['ConstantDecoder'], 'ConstantDecoder')],
        'AggregatorDecoder': [],
        'RootDecoder': [(decoder_dict['AggregatorDecoder'], 'AggregatorDecoder'),
                        (decoder_dict['ColumnDecoder'], 'ColumnDecoder'),
                        (decoder_dict['TableDecoder'], 'TableDecoder'),
                        (decoder_dict['KeywordDecoder'], 'KeywordDecoder'),
                        ],
        'AndOrDecoder': [(decoder_dict['ColumnDecoder'], 'ColumnDecoder')],
        'ConstantDecoder': [(decoder_dict['AndOrDecoder'], 'AndOrDecoder')]
    }

    return switch.get(current_decoder, "None")


def get_decoder_vocab_dicts(global_dict, table_dict, col_dict):
    """
    Makes a dictionary of vocab dictionaries for each decoder
    Args:
        global_dict (dict): Full vocab dictionary
        table_dict (dict): Dictionary of table names
        col_dict (dict): Dictionary of column names
    Returns: Dictionary of vocab dictionaries for each decoder
    """

    keyword_dict = {
        'where': 0,
        '<none>': 1
    }
    operator_dict = {
        '>': 0,
        '<': 1,
        '=': 2,
        '<none>': 3
    }
    aggregator_dict = {
        'min': 0,
        'max': 1,
        'sum': 2,
        'avg': 3,
        'count': 4,
        '<none>': 5
    }
    and_or_dict = {
        'and': 0,
        '<none>': 1,
    }

    dict_ = {  # Dictionary with next model calls in reverse order
        'KeywordDecoder': keyword_dict,
        'ColumnDecoder': col_dict,
        'TableDecoder': table_dict,
        'OperatorDecoder': operator_dict,
        'AggregatorDecoder': aggregator_dict,
        'RootDecoder': {},
        'AndOrDecoder': and_or_dict,
        'ConstantDecoder': global_dict
    }

    return dict_


def save_models(encoder_dict, decoder_dict, epoch):
    """
    Saves all encoder and decoder models
    Args:
        encoder_dict (dict): Dictionary of all encoder objects
        decoder_dict (dict): Dictionary of all decoder objects
        epoch (int): Epoch number
    Returns: None
    """
    try:
        os.mkdir('saved_models/' + str(epoch))
        os.mkdir('saved_models/' + str(epoch) + '/encoders')
        os.mkdir('saved_models/' + str(epoch) + '/decoders')
    except OSError:
        pass
    for name, encoder in encoder_dict.items():
        torch.save(encoder.state_dict(), 'saved_models/' + str(epoch) + '/encoders/' + str(name) + '.pt')
    for name, decoder in decoder_dict.items():
        torch.save(decoder.state_dict(), 'saved_models/' + str(epoch) + '/decoders/' + str(name) + '.pt')


def load_models(encoder_dict, decoder_dict, epoch):
    """
    Loads all encoder and decoder models
    Args:
        encoder_dict (dict): Dictionary of all encoder objects
        decoder_dict (dict): Dictionary of all decoder objects
        epoch (int): Epoch to be loaded
    Returns: None
    """
    for name, encoder in encoder_dict.items():
        encoder.load_state_dict(torch.load('saved_models/' + str(epoch) + '/encoders/' + str(name) + '.pt'))
    for name, decoder in decoder_dict.items():
        decoder.load_state_dict(torch.load('saved_models/' + str(epoch) + '/decoders/' + str(name) + '.pt'))


def zero_all_grads(encoder_optimizer, decoder_optimizer):
    """
    Zeros gradients for all encoder and decoder parameters
    Args:
        encoder_optimizer (dict): Dictionary of all encoder optimizers
        decoder_optimizer (dict): Dictionary of all decoder optimizers
    Returns: None
    """
    for encoder in encoder_optimizer.values():
        encoder.zero_grad()
    for decoder in decoder_optimizer.values():
        decoder.zero_grad()
        
def save_pickle(file, filename):
    """
    Args:
        file: file to be saved
        filename (str): path of the file to be saved
    Returns: None
    """
    with open(filename, 'wb') as f:
        pickle.dump(file, f)


def load_pickle(filename):
    """
    Args:
        filename (str): name of the pickle file to be loaded
    Returns:
        file: data loaded from pickle
    """
    with open(filename, 'rb') as f:
        file = pickle.load(f)
    return file
