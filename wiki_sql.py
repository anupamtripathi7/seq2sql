import torch
from torch.utils.data import Dataset
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

root = ''
split = 0.8


class WikiSQL(Dataset):

    def __init__(self, text, sql, schema, transform=None):
        """
        Args:
            text (string): File location of text to be converted to sql
            sql (string): File location of corresponding sql queries
            schema (string): File location of corresponding database schema
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        with open(text, 'rb') as f:
            self.text = pickle.load(f, encoding='bytes')
        with open(sql, 'rb') as file_b:
            self.sql = pickle.load(file_b, encoding='bytes')
        with open(schema, 'rb') as file_b:
            self.schema = pickle.load(file_b, encoding='bytes')

        self.max_len_text = 0
        self.max_len_sql = 0
        self.max_len_schema = 0

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        """
        Args:
            idx (Tensor): Indices of the data to be returned
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = torch.from_numpy(self.text[idx]).float()
        sql = torch.from_numpy(self.sql[idx]).float()
        schema = torch.from_numpy(self.schema[idx]).float()

        self.max_len_text = max(self.max_len_text, len(text))
        self.max_len_sql = max(self.max_len_sql, len(sql))
        self.max_len_schema = max(self.max_len_schema, len(schema))

        # if self.transform:
        #     sample = self.transform(sample)
        return [text, sql, schema]

    def collate(self, batch):
        """
        Called for each batch for adding padding
        Args:
            batch (Tensor): Batch of data
        Returns:
            text (Tensor): Text data
            sql (Tensor): Query data
            schema (Tensor): Schema data
        """
        text = []
        sql = []
        schema = []
        for n, x in enumerate(batch):
            text.append(torch.cat((x[0], torch.zeros(self.max_len_text - x[0].size(0))), dim=0))
            sql.append(torch.cat((x[1], torch.zeros(self.max_len_sql - x[1].size(0))), dim=0))
            schema.append(torch.cat((x[2], torch.zeros(self.max_len_schema - x[2].size(0))), dim=0))
        self.max_len_text, self.max_len_sql, self.max_len_schema = 0, 0, 0

        return torch.stack(text), torch.stack(sql), torch.stack(schema)

# if __name__ == '__main__':
#
#     questions_path = 'data/questions/'
#     sql_queries_path = 'data/sql_queries/'
#     word_idx_mappings_path = 'data/word_idx_mappings/'
#     wiki_sql_path = 'data/WikiSQL_files/'
#     compose = transforms.Compose(
#         [transforms.ToTensor(),
#          ])
#     transformed_dataset = WikiSQL(text=os.path.join(questions_path, 'train_questions_tokenized.pkl'),
#                                   sql=os.path.join(sql_queries_path, 'train_sql_tokenized.pkl'),
#                                   transform=compose)
#
#     dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=True)
#
#     for x in dataloader:
#         print(x['text'])
#         print(x['sql'])
#         print(torch.unique(x['train_text']))
