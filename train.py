import torch
from torch import optim
import torch.nn as nn
from wiki_sql import WikiSQL
from model import Encoder, Decoder
from torch.utils.data import DataLoader
from utils import load_pickle
import os
from tqdm import tqdm
from utils import get_decoder_vocab_dicts, save_models, zero_all_grads
from extract_data import load_pickle
import os
import matplotlib.pyplot as plt

root = 'data'
# questions_path = 'data/questions/'
# sql_queries_path = 'data/sql_queries/'
# word_idx_mappings_path = 'data/word_idx_mappings/'
# wiki_sql_path = 'data/WikiSQL_files/'
vocab_size = 1                                                                                                          # Size of vocab, set later
enc_hidden_size = 40                                                                                                    # Size of h from each LSTM cell encoder        2*enc > dec
dec_hidden_size = 30                                                                                                    # Should be even
num_layers = 1                                                                                                          # Number of LSTM cells stacked one above other. Not used
num_epochs = 10
learning_rate = 3e-5
sequence_length = 1                                                                                                     # One word per lstm cell (not used)
encoder_output_size = 0                                                                                                 # Size of encoding and size of decoder input
batch_size = 1
embed_dim = 50
# keyword_output_dim = 2                                                                                                  # [where, none]
# column_output_dim = 0                                                                                                   # no of col in vocab, set later
# table_output_dim = 0                                                                                                    # no of tables in vocab, set later
# operator_output_dim = 6                                                                                                 # {>, <, =, <=, >=, !=, none}
# aggregator_output_dim = 6                                                                                               # {max, min, count, sum, avg, none}
# # root_output_dim = 0                                                                                                   # no of tables in vocab, set later
# and_or_output_dim = 3                                                                                                   # {and, or, none}
# constant_output_dim = 0                                                                                                 # V + col. Set later


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    train_transformed_dataset = WikiSQL(text=os.path.join(root, 'train/train_questions_tokenized.pkl'),
                                        sql=os.path.join(root, 'train/train_sql_tokenized.pkl'),
                                        schema=os.path.join(root, 'train/', 'train_schema.pkl')
                                        )

    test_transformed_dataset = WikiSQL(text=os.path.join(root, 'test/test_questions_tokenized.pkl'),
                                        sql=os.path.join(root, 'test/test_sql_tokenized.pkl'),
                                       schema=os.path.join(root, 'test/test_schema.pkl')
                                       )

    valid_transformed_dataset = WikiSQL(text=os.path.join(root, 'valid/valid_questions_tokenized.pkl'),
                                        sql=os.path.join(root, 'valid/valid_sql_tokenized.pkl'),
                                        schema=os.path.join(root, 'valid/valid_schema.pkl')
                                        )

    word2idx = load_pickle(os.path.join(root, 'word_idx_mappings/word2idx.pkl'))
    idx2word = load_pickle(os.path.join(root, 'word_idx_mappings/idx2word.pkl'))
    col_dict = load_pickle(os.path.join(root, 'word_idx_mappings/column_mappings1.pkl'))
    table_dict = load_pickle(os.path.join(root, 'word_idx_mappings/table_mappings1.pkl'))
    vocab_size = len(word2idx.keys())

    vocab_dicts = get_decoder_vocab_dicts(word2idx, table_dict, col_dict)

    train_loader = DataLoader(train_transformed_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=train_transformed_dataset.collate)
    test_loader = DataLoader(test_transformed_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=test_transformed_dataset.collate)
    valid_loader = DataLoader(valid_transformed_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=valid_transformed_dataset.collate)
    loss = nn.NLLLoss()

    # emb_dim, hidden_size, decoder_hidden_size, vocab_size
    encoder_dict = {
        'QuestionEncoder': Encoder(embed_dim, enc_hidden_size, int(dec_hidden_size / 2), vocab_size).to(device),
        'SchemaEncoder': Encoder(embed_dim, enc_hidden_size, int(dec_hidden_size / 2), vocab_size).to(device)
    }

    # emb_dim, vocab_size, hidden_size, encoder_hidden_size, output_dim, decoder_vocab_dict, global_dict
    decoder_dict = {
        'KeywordDecoder': Decoder(embed_dim, vocab_size, dec_hidden_size, enc_hidden_size, vocab_dicts['KeywordDecoder'], idx2word).to(device),
        'ColumnDecoder': Decoder(embed_dim, vocab_size, dec_hidden_size, enc_hidden_size, vocab_dicts['ColumnDecoder'], idx2word).to(device),
        'TableDecoder': Decoder(embed_dim, vocab_size, dec_hidden_size, enc_hidden_size, vocab_dicts['TableDecoder'], idx2word).to(device),
        'OperatorDecoder': Decoder(embed_dim, vocab_size, dec_hidden_size, enc_hidden_size, vocab_dicts['OperatorDecoder'], idx2word).to(device),
        'AggregatorDecoder': Decoder(embed_dim, vocab_size, dec_hidden_size, enc_hidden_size, vocab_dicts['AggregatorDecoder'], idx2word).to(device),
        'RootDecoder': Decoder(embed_dim, vocab_size, dec_hidden_size, enc_hidden_size, vocab_dicts['RootDecoder'], idx2word).to(device),
        'AndOrDecoder': Decoder(embed_dim, vocab_size, dec_hidden_size, enc_hidden_size, vocab_dicts['AndOrDecoder'], idx2word).to(device),
        'ConstantDecoder': Decoder(embed_dim, vocab_size, dec_hidden_size, enc_hidden_size, vocab_dicts['ConstantDecoder'], idx2word).to(device)
    }

    encoder_optimizer, decoder_optimizer = {}, {}
    for name, encoder in encoder_dict.items():
        encoder_optimizer[name] = optim.Adam(encoder.parameters(), lr=learning_rate)
    for name, decoder in decoder_dict.items():
        decoder_optimizer[name] = optim.Adam(decoder.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    losses = []
    for epoch in range(num_epochs):
        total_cost = 0
        for n, sample in enumerate(tqdm(train_loader)):

            text = sample[0].to(device)
            sql = sample[1].to(device)
            schema = sample[2].to(device)

            question_enc_outputs, question_enc_hidden = encoder_dict['QuestionEncoder'](text.t())
            question_enc_outputs = question_enc_outputs.to(device)
            question_enc_hidden = question_enc_hidden.to(device)
            schema_enc_outputs, schema_enc_hidden = encoder_dict['SchemaEncoder'](schema.t())
            schema_enc_outputs = schema_enc_outputs.to(device)
            schema_enc_hidden = schema_enc_hidden.to(device)

            hidden = torch.cat((question_enc_hidden, schema_enc_hidden), dim=-1)
            enc_outputs = torch.cat((question_enc_outputs, schema_enc_outputs), dim=0)

            if n in [0, int(len(train_loader) / 3), 2 * int(len(train_loader) / 3)]:
                print_outputs = True
                print('\n\nText: {}\nSQL: {}'.format(' '.join(list(map(lambda i: idx2word[i.item()], text[0]))),
                                                     ' '.join(list(map(lambda i: idx2word[i.item()], sql[0])))))
            else:
                print_outputs = False

            # sql, idx, h, encoder_outputs, decoder_dict, current_decoder, predictions, print_outputs
            outputs, targets, decoders = decoder_dict['RootDecoder'](sql, 0, hidden, enc_outputs, decoder_dict, 'RootDecoder', print_output=print_outputs)

            epoch_loss = 0
            for output, target, decoder_name in zip(outputs, targets, decoders):
                zero_all_grads(encoder_optimizer, decoder_optimizer)
                cost = loss(output, target.to(device))
                cost.backward(retain_graph=True)
                decoder_optimizer[decoder_name].step()
                epoch_loss += cost

            zero_all_grads(encoder_optimizer, decoder_optimizer)                                                        # Zeros grads for all encoders
            epoch_loss.backward()                                                                                       # Backward for encoder loss
            for encoder in encoder_optimizer.values():
                encoder.step()

            total_cost += epoch_loss.item()

        valid_loss = 0
        with torch.no_grad():
            for n, sample in enumerate(tqdm(valid_loader)):

                text = sample[0].to(device)
                sql = sample[1].to(device)
                schema = sample[2].to(device)

                question_enc_outputs, question_enc_hidden = encoder_dict['QuestionEncoder'](text.t())
                question_enc_outputs = question_enc_outputs.to(device)
                question_enc_hidden = question_enc_hidden.to(device)
                schema_enc_outputs, schema_enc_hidden = encoder_dict['SchemaEncoder'](schema.t())
                schema_enc_outputs = schema_enc_outputs.to(device)
                schema_enc_hidden = schema_enc_hidden.to(device)

                hidden = torch.cat((question_enc_hidden, schema_enc_hidden), dim=-1)
                enc_outputs = torch.cat((question_enc_outputs, schema_enc_outputs), dim=0)

                if n in [0, int(len(train_loader) / 3), 2 * int(len(train_loader) / 3)]:
                    print_outputs = True
                    print('\n\nText: {}\nSQL: {}'.format(' '.join(list(map(lambda i: idx2word[i.item()], text[0]))),
                                                         ' '.join(list(map(lambda i: idx2word[i.item()], sql[0])))))
                else:
                    print_outputs = False

                # sql, idx, h, encoder_outputs, decoder_dict, current_decoder, predictions, print_outputs
                outputs, targets, decoders = decoder_dict['RootDecoder'](sql, 0, hidden, enc_outputs, decoder_dict, 'RootDecoder', print_output=print_outputs)

                epoch_loss = 0
                for output, target, decoder_name in zip(outputs, targets, decoders):
                    zero_all_grads(encoder_optimizer, decoder_optimizer)
                    epoch_loss += loss(output, target)

                valid_loss += epoch_loss.double()
        print('Epoch: {}    total_cost =  {}    validLoss = {}'.format(epoch, total_cost, valid_loss))
        save_models(encoder_dict, decoder_dict, epoch)

        losses.append([total_cost, valid_loss])

    plt.plot(losses)
    plt.show()
