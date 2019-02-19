import torch
from torch import  nn
from torch.nn import init

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers=1, dropout=0):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size= hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Construct embedding matrix
        self.embedding = nn.Embedding(self.vocab_size, self.input_size, padding_idx=0)

        # Construct RNN cell
        self.rnn = nn.GRU(input_size= self.input_size, hidden_size= self.hidden_size, num_layers=self.hidden_size)

        self.initialize_weights()


    def initialize_weights(self):

        import pdb
        pdb.set_trace()

        # initialize RNN weights
        for cur_layer in self.rnn._all_weights:
            for p in cur_layer:
                if 'weight' in p:
                    init.orthogonal(self.rnn.__getattr__(p))

        # Initialze embedding vector
        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
        init.uniform(embedding_weights, a= -0.25, b=0.25)

        # padding
        embedding_weights[self.PAD] = torch.FloatTensor([0]*self.input_size)

        # Assign embedding vector
        del self.embedding.weight
        self.embedding.weight = nn.Parameter(embedding_weights)


    def forward(self, *input):
        embs = self.embedding(input)
        output, hidden = self.rnn(embs)
        return output, hidden


class DualEncoder(nn.Module):
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()

        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size

        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal(M)
        self.M = nn.Parameter(M, requires_grad = True)


    def forward(self, message_tensor, response_tensor):
        _, message_emb = self.encoder(message_tensor)
        _, reply_emb = self.encoder(response_tensor)

        context = message_emb.mm(self.M)
        context = context.view(-1, 1, self.hidden_size)

        response = reply_emb.view(-1, self.hidden_size, 1)

        score = torch.bmm(context, response).view(-1, 1)

        return score










encoder_model = Encoder(
  input_size=100, # embedding dim
  hidden_size=300, # rnn dim
  vocab_size=91620
)

import pdb
pdb.set_trace()
encoder_model.cuda()
