# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from base.base_model import BaseModel

# 自定义可设置维度的softmax 源码地址：https://discuss.pytorch.org/t/why-softmax-function-cant-specify-the-dimension-to-operate/2637
def softmax(input, axis=1):
    input_size = input.size()
    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()
    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)


def softmax_mask(A, mask, dim=1, epsilon=1e-12):
    # According to https://discuss.pytorch.org/t/apply-mask-softmax/14212/7
    A_max,_ = torch.max(A,dim=dim,keepdim=True)
    A_exp = torch.exp(A-A_max)
    A_exp = A_exp * mask # this step masks
    A_softmax = A_exp/(torch.sum(A_exp,dim=dim,keepdim=True)+epsilon)
    return A_softmax


def sort_batch(data, seq_len, cuda_able):
    sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
    sorted_data = data[sorted_idx.data]
    _, reverse_idx = torch.sort(sorted_idx, dim=0, descending=False)
    if cuda_able:
        sorted_seq_len, reverse_idx = sorted_seq_len.cuda(),  reverse_idx.cuda()
    return sorted_data, sorted_seq_len, reverse_idx


class WikiqaModel(BaseModel):
    def __init__(self, embedding_matrix, rnn_hidden, rnn_layers, linear_hidden,
                 bidirectional, cuda_able):
        super(WikiqaModel, self).__init__()
        vocab_size = embedding_matrix.size(0)
        embedding_size = embedding_matrix.size(1)
        self.cuda_able = cuda_able
        self.rnn_layers = rnn_layers
        self.rnn_hidden = rnn_hidden
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding.weight = nn.Parameter(embedding_matrix)  # 使用预训练词向量
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=rnn_hidden, num_layers=self.rnn_layers,
                            batch_first=True, bidirectional=bidirectional)

        self.maxpool = nn.AdaptiveMaxPool2d((1, self.num_directions * rnn_hidden))

        self.fc1 = nn.Linear(self.num_directions * rnn_hidden, linear_hidden)

        self.W = nn.parameter.Parameter(
            torch.Tensor(rnn_hidden * self.num_directions, rnn_hidden * self.num_directions).uniform_(-0.05, 0.05))

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(self.num_directions * self.rnn_layers, batch_size, self.rnn_hidden)
        c_0 = torch.zeros(self.num_directions * self.rnn_layers, batch_size, self.rnn_hidden)
        return (Variable(h_0.cuda()), Variable(c_0.cuda())) if self.cuda_able else(Variable(h_0), Variable(c_0))

    def get_encode(self, s, s_len):
        ############# with pack_pad
        # 参考：https://github.com/kevinkwl/AoAReader
        # Note: There is a subtlety in using the pack sequence -> recurrent network -> unpack sequence pattern in a Module wrapped in DataParallel,
        #       see https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism for detail
        s_sorted, s_len_sorted, s_idx_reverse = sort_batch(s, s_len, self.cuda_able)
        total_length = s_sorted.size(1)

        s_embedding = self.embedding(s_sorted)
        s_embedding = pack(s_embedding, s_len_sorted, batch_first=True)
        # encode
        s_output, _ = self.lstm(s_embedding, self.hidden)
        # unpack
        s_output, _ = unpack(s_output, batch_first=True, total_length=total_length)
        s_output = s_output[s_idx_reverse.data]
        ############# without pack_pad
        # s_embedding = self.get_sentence_embedding(s, overlap)
        # s_output, _ = self.lstm(s_embedding, self.hidden)
        return s_output

    def get_sim_score(self, q, q_mask, q_len, a, a_mask, a_len):
        # 计算问题和答案之间的相似度

        # Encode
        a = self.get_encode(a, a_len)
        q = self.get_encode(q, q_len)

        q_mask = q_mask.unsqueeze(2)
        a_mask = a_mask.unsqueeze(2)

        # Attention Between question and answer
        attn_mat = torch.matmul(
            torch.matmul(a, self.W),
            q.transpose(1, 2))
        weight_a = softmax_mask(torch.max(attn_mat, 2, keepdim=True)[0], a_mask, dim=1)
        weight_q = softmax_mask(torch.max(attn_mat, 1, keepdim=True)[0].transpose(1, 2), q_mask, dim=1)
        a = weight_a * a
        q = weight_q * q

        # Pooling Layer
        a = self.maxpool(a)
        q = self.maxpool(q)

        # FNN
        a = a.view(a.size(0), -1)
        q = q.view(q.size(0), -1)
        # concat hypernym feature
        # if self.hypernym:
        #     a_hypernym = arr[2] if self.wordoverlap else arr[0]
        #     q_hypernym = arr[3] if self.wordoverlap else arr[1]
        #
        #     a_hypernym = a_hypernym.view(a_hypernym.size(0), -1)
        #     q_hypernym = Variable(torch.zeros(batch_size, 1).cuda()) if self.cuda_able else  Variable(
        #         torch.zeros(batch_size, 1))
        #
        #     a = torch.cat((a, a_hypernym), 1)
        #     q = torch.cat((q, q_hypernym), 1)

        a = torch.tanh(self.fc1(a))
        q = torch.tanh(self.fc1(q))
        # get consine similarity
        score = F.cosine_similarity(a, q)
        return score, weight_a, weight_q

    def forward(self, data):
        # wa/q/ra 分别表示错误答案，问题，正确答案

        wa = data['wrong_answer']
        wa_mask = data['wrong_answer_mask']
        wa_len = data['wrong_answer_len']
        q = data['question']
        q_mask = data['question_mask']
        q_len = data['question_len']
        ra = data['right_answer']
        ra_mask = data['right_answer_mask']
        ra_len = data['right_answer_len']

        # for key in data:
        #     print(key, data[key].size())

        batch_size = q.size(0)
        self.hidden = self.init_hidden(batch_size)

        pos_cosine, weight_ra, weight_q_ra = self.get_sim_score(q, q_mask, q_len, ra, ra_mask, ra_len)
        if data['predict']:
            neg_cosine, weight_wa, weight_q_wa = None, None, None
        else:
            neg_cosine, weight_wa, weight_q_wa = self.get_sim_score(q, q_mask, q_len, wa, wa_mask, wa_len)
        # print(type(pos_cosine), type(neg_cosine))
        return (pos_cosine, neg_cosine)
