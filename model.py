import torch
import torch.nn as nn
from utils import get_next_decoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, emb_dim, hidden_size, decoder_hidden_size, vocab_size):
        """
        Args:
            emb_dim (int): Embedding size
            hidden_size (int): Encoder hidden size
            decoder_hidden_size (int): Decoder hidden size
            vocab_size (int): Size of vocab
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_size, bidirectional=True, dropout=0.3)

        self.fc = nn.Linear(hidden_size * 2, decoder_hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Args:
            x(tensor): Input sentence of size (src len, batch size)
        Returns: Encoder output (src len, batch size, hidden * 2) and hidden (batch size, decoder_hidden_size)
        """
        x = self.dropout(self.embedding(x.long()))                                                                      # (src len, batch size, emb_dim)
        x, h = self.gru(x)
        h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)                                                                # concat last forward and backward cell's outputs
        h = torch.tanh(self.fc(h))
        return x, h


class Decoder(nn.Module):

    def __init__(self, emb_dim, vocab_size, hidden_size, encoder_hidden_size, decoder_vocab_dict, global_dict):
        """
        Args:
            emb_dim (int): Embedding size
            vocab_size (int): Size of vocab
            hidden_size (int): Decoder hidden size
            encoder_hidden_size (int): Encoder hidden size
            decoder_vocab_dict (dict): Dictionary of decoder vocab
            global_dict (dict): Dictionary of full vocab
        """
        super(Decoder, self).__init__()
        output_dim = len(decoder_vocab_dict.keys())
        self.hidden_size = hidden_size
        self.decoder_vocab_dict = decoder_vocab_dict
        self.global_dict = global_dict

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU((encoder_hidden_size * 2) + emb_dim, hidden_size, dropout=0.3)

        self.attn_fc = nn.Linear((encoder_hidden_size * 2) + hidden_size, 1)

        self.fc = nn.Linear((encoder_hidden_size * 2) + hidden_size + emb_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sql, idx, h, encoder_outputs, decoder_dict, current_decoder, predictions=None, targets=None,
                decoders=None, print_output=False):
        """
        Args:
            sql (tensor): Tensor of all sql (batch_size, src len)
            idx (int): Index of current input
            h (tensor): (batch_size, hidden_size)
            encoder_outputs (tensor): (src len, batch size, enc_hid_dim * 2)
            decoder_dict (dict): Dictionary of decoder models
            current_decoder (str): Current decoder name
            predictions (list): List of all predictions
            targets (list): List of targets in local dictionary mappings
            decoders (list): List of decoders for each prediction
            print_output (bool): If true, give output on screen
        Returns: Decoder output (batch size, vocab) and hidden (1, batch size, hidden)
        """
        if decoders is None:
            decoders, targets, predictions = [], [], []
        if print_output:
            print('\n\nCurrent Decoder: {}\ni = {}'.format(current_decoder, idx))
        if current_decoder == 'RootDecoder':
            next_decoders = get_next_decoder(current_decoder, decoder_dict)

            for next_decoder, decoder_name in next_decoders:
                idx += 1
                if decoder_name == 'AggregatorDecoder':
                    # predictions.append(torch.tensor([26173]))                                                         # select
                    # targets.append(torch.tensor([26173]))
                    idx += 1
                elif decoder_name == 'TableDecoder':
                    # predictions.append(torch.tensor([19884]))                                                         # from
                    # targets.append(torch.tensor([19884]))
                    idx += 1
                predictions, targets, decoders, idx = next_decoder(sql, idx, h, encoder_outputs, decoder_dict,
                                                                   decoder_name, predictions, targets, decoders,
                                                                   print_output)

                if idx == sql.size(1):
                    return predictions, targets, decoders

            if idx != sql.size(1):
                predictions.extend([predictions[-1] for x in range(sql.size(1) - idx)])                                 # appending padding of 1s to punish decoder for not predicting the remaining words
                targets.extend([targets[-1] for x in range(sql.size(1) - idx)])                                         # appending padding of 0s
                decoders.extend([decoders[-1] for x in range(sql.size(1) - idx)])

            return predictions, targets, decoders

        if idx == sql.size(1):
            return predictions, targets, decoders, idx
        x = sql[:, idx]

        # Attention
        src_len = encoder_outputs.shape[0]
        h_rep = h.unsqueeze(1).repeat(1, src_len, 1)                                                                    # (batch size, src len, dec hid dim) 5, 100, 20
        encoder_outputs_rearranged = encoder_outputs.permute(1, 0, 2)                                                   # (batch size, src len, enc_hid_dim * 2) 5, 100, 100
        weights = torch.tanh(self.attn_fc(torch.cat((h_rep, encoder_outputs_rearranged), dim=2)))                       # (batch size, src len, 1)

        embedded = self.embedding(x.long()).squeeze(1)                                                                  # (batch_size, emb_dim)
        embedded = self.dropout(embedded)

        weighted = torch.bmm(encoder_outputs_rearranged.permute(0, 2, 1), weights).squeeze(2)                           # (batch size, enc_hid_dim * 2)
        x = torch.cat((embedded, weighted), dim=1)                                                                      # (batch size, emb_dim + enc_hid_dim*2)
        x, h = self.gru(x.unsqueeze(0), h.unsqueeze(0))                                                                 # src size = 1

        # assert (x == h).all()

        x = self.softmax(self.fc(torch.cat((x.squeeze(0), weighted, embedded), dim=1)))                                 # (batch size, output_dim)
        # x = x.argmax(1)                                                                                               # (batch_size, 1)
        # for n, index in enumerate(x):                                                                                   # Decoder vocab to word mapping
        #     print(self.decoder_vocab_dict[index.item()])
        #     x[n] = self.global_dict[self.decoder_vocab_dict[index.item()]]
        if x.argmax(1) != x.size(1) - 1 or current_decoder not in ['KeywordDecoder', 'OperatorDecoder', 'AndOrDecoder',
                                                                   'AggregatorDecoder']:
            if print_output:
                print('Predicted: {}\nTarget: {}'.format(list(self.decoder_vocab_dict.keys())[x.argmax(1).item()],
                                                               self.global_dict[int(sql[:, idx].item())]))
            predictions.append(x)
            try:
                targets.append(torch.tensor([self.decoder_vocab_dict[self.global_dict[sql[:, idx].item()]]]))
            except KeyError:
                targets.append(torch.tensor([x.size(1) - 1]))
            decoders.append(current_decoder)

            next_decoders = get_next_decoder(current_decoder, decoder_dict)

            if self.global_dict[int(sql[:, idx].item())] not in self.decoder_vocab_dict.keys():                         # Wrong decoder
                idx -= 1
                return predictions, targets, decoders, idx
            for next_decoder, decoder_name in next_decoders:
                idx = idx + 1
                predictions, targets, decoders, idx = next_decoder(sql, idx, h.squeeze(0), encoder_outputs,
                                                                   decoder_dict, decoder_name, predictions,
                                                                   targets, decoders, print_output)
        else:                                                                                                           # None
            if self.global_dict[int(sql[:, idx].item())] in self.decoder_vocab_dict.keys():                             # Wrongly predicted None
                predictions.append(x)
                try:
                    targets.append(torch.tensor([self.decoder_vocab_dict[self.global_dict[sql[:, idx].item()]]]))
                except KeyError:
                    targets.append(torch.tensor([x.size(1) - 2]))
                decoders.append(current_decoder)
            else:
                idx -= 1

        return predictions, targets, decoders, idx


# if __name__ == "__main__":
#     encoder = Encoder(3, 50, 40, 1000)
#     inp = torch.randint(0, 1000, (100, 5))
#     output = encoder(inp)
#     print(output[0].size(), output[1].size())
#
#     decoder = Decoder(3, 1000, 40, 50)
#     print(decoder(torch.randint(0, 1000, (5, 1)), output[1], output[0]))
