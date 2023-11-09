import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy.stats.stats as stats
import numpy as np

def get_rank(batch_score, dim=0):
    rank = torch.argsort(batch_score, dim=dim)
    rank = torch.argsort(rank, dim=dim)
    rank = (rank * -1) + batch_score.size(dim)
    rank = rank.float()
    rank = rank / batch_score.size(dim)

    return rank

def get_tiedrank(batch_score, dim=0):
    batch_score = batch_score.cpu()
    rank = stats.rankdata(batch_score)
    rank = stats.rankdata(rank) - 1
    rank = (rank * -1) + batch_score.size(dim)
    rank = torch.from_numpy(rank).cuda()
    rank = rank.float()
    rank = rank / batch_score.size(dim)
    return rank

def model_loader(model_type, seq_len, pretrained_state_dict=None):

    if model_type == "lstm":
        model = lstm_baseline(seq_len)
    elif model_type == "grus":
        model = gru_sum(seq_len)
    elif model_type == "gruc":
        model = gru_constrained(seq_len)
    elif model_type == "grup":
        model = gru_proj(seq_len)
    elif model_type == "exa":
        model = sorter_exact()
    elif model_type == "lstmla":
        model = lstm_large(seq_len)
    elif model_type == "lstme":
        model = lstm_end(seq_len)
    elif model_type == "mlp":
        model = mlp(seq_len)
    elif model_type == "cnn":
        return cnn(seq_len)
    else:
        raise Exception("Model type unknown", model_type)

    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)

    return model

class lstm_baseline(nn.Module):
    def __init__(self, seq_len):
        super(lstm_baseline, self).__init__()
        self.lstm = nn.LSTM(1, 128, 2, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(seq_len, seq_len, 256)

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1, 1)
        out, _ = self.lstm(input_)
        out = self.conv1(out)

        return out.view(input_.size(0), -1)


class gru_constrained(nn.Module):
    def __init__(self, seq_len):
        super(gru_constrained, self).__init__()
        self.rnn = nn.GRU(1, 32, 6, batch_first=True, bidirectional=True)

        self.sig = torch.nn.Sigmoid()

    def forward(self, input_):
        input_ = (input_.reshape(input_.size(0), -1, 1) / 2.0) + 1
        input_ = self.sig(input_)

        x, hn = self.rnn(input_)
        out = x.sum(dim=2)

        out = self.sig(out)

        return out.view(input_.size(0), -1)


class gru_proj(nn.Module):

    def __init__(self, seq_len):
        super(gru_proj, self).__init__()
        self.rnn = nn.GRU(1, 128, 6, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(seq_len, seq_len, 256)

        self.sig = torch.nn.Sigmoid()

    def forward(self, input_):
        input_ = (input_.reshape(input_.size(0), -1, 1) / 2.0) + 1

        input_ = self.sig(input_)

        out, _ = self.rnn(input_)
        out = self.conv1(out)

        out = self.sig(out)

        return out.view(input_.size(0), -1)


class cnn(nn.Module):
    def __init__(self, seq_len):
        super(cnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, 2),
            nn.PReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, 3),
            nn.BatchNorm1d(16),
            nn.PReLU())
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, 5),
            nn.PReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64, 7),
            nn.BatchNorm1d(64),
            nn.PReLU())
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 96, 10),
            nn.PReLU())
        self.layer6 = nn.Sequential(
            nn.Conv1d(96, 128, 7),
            nn.BatchNorm1d(128),
            nn.PReLU())
        self.layer7 = nn.Sequential(
            nn.Conv1d(128, 256, 5),
            nn.PReLU())
        self.layer8 = nn.Sequential(
            nn.Conv1d(256, 256, 3),
            nn.BatchNorm1d(256),
            nn.PReLU())
        self.layer9 = nn.Sequential(
            nn.Conv1d(256, 128, 3),
            nn.PReLU())
        self.layer10 = nn.Conv1d(128, seq_len, 64)

    def forward(self, input_):
        out = input_.unsqueeze(1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out).view(input_.size(0), -1)
        out = torch.sigmoid(out)

        out = out
        return out


class mlp(nn.Module):
    def __init__(self, seq_len):
        super(mlp, self).__init__()
        self.lin1 = nn.Linear(seq_len, 2048)
        self.lin2 = nn.Linear(2048, 2048)
        self.lin3 = nn.Linear(2048, seq_len)

        self.relu = nn.ReLU()

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1)
        out = self.lin1(input_)
        out = self.lin2(self.relu(out))
        out = self.lin3(self.relu(out))

        return out.view(input_.size(0), -1)


class gru_sum(nn.Module):
    def __init__(self, seq_len):
        super(gru_sum, self).__init__()
        self.lstm = nn.GRU(1, 4, 1, batch_first=True, bidirectional=True)

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1, 1)
        out, _ = self.lstm(input_)
        out = out.sum(dim=2)

        return out.view(input_.size(0), -1)


class lstm_end(nn.Module):
    def __init__(self, seq_len):
        super(lstm_end, self).__init__()
        self.seq_len = seq_len
        self.lstm = nn.GRU(self.seq_len, 5 * self.seq_len, batch_first=True, bidirectional=False)

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1, 1).repeat(1, input_.size(1), 1).view(input_.size(0), input_.size(1), -1)
        _, out = self.lstm(input_)

        out = out.view(input_.size(0), self.seq_len, -1)  # .view(input_.size(0), -1)[:,:self.seq_len]
        out = out.sum(dim=2)

        return out

class sorter_exact(nn.Module):

    def __init__(self):
        super(sorter_exact, self).__init__()

    def comp(self, inpu):
        in_mat1 = torch.triu(inpu.repeat(inpu.size(0), 1), diagonal=1)
        in_mat2 = torch.triu(inpu.repeat(inpu.size(0), 1).t(), diagonal=1)

        comp_first = (in_mat1 - in_mat2)
        comp_second = (in_mat2 - in_mat1)

        std1 = torch.std(comp_first).item()
        std2 = torch.std(comp_second).item()

        comp_first = torch.sigmoid(comp_first * (6.8 / std1))
        comp_second = torch.sigmoid(comp_second * (6.8 / std2))

        comp_first = torch.triu(comp_first, diagonal=1)
        comp_second = torch.triu(comp_second, diagonal=1)

        return (torch.sum(comp_first, 1) + torch.sum(comp_second, 0) + 1) / inpu.size(0)

    def forward(self, input_):
        out = [self.comp(input_[d]) for d in range(input_.size(0))]
        out = torch.stack(out)

        return out.view(input_.size(0), -1)


class lstm_large(nn.Module):

    def __init__(self, seq_len):
        super(lstm_large, self).__init__()
        self.lstm = nn.LSTM(1, 512, 2, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(seq_len, seq_len, 1024)

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1, 1)
        out, _ = self.lstm(input_)
        out = self.conv1(out)

        return out.view(input_.size(0), -1)

# model_type, seq_len, checkpoint_path
def load_sorter(checkpoint_path):
    sorter_checkpoint = torch.load(checkpoint_path)

    model_type = sorter_checkpoint["args_dict"].model_type ##
    seq_len = sorter_checkpoint["args_dict"].seq_len ##
    state_dict = sorter_checkpoint["state_dict"]

    return model_type, seq_len, state_dict

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

class SpearmanLoss(torch.nn.Module):
    """ Loss function  inspired by spearmann correlation.self
    Required the trained model to have a good initlization.

    Set lbd to 1 for a few epoch to help with the initialization.
    """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None, lbd=1):
        super(SpearmanLoss, self).__init__()
        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict) ##
        # self.seq_len = seq_len
        # self.sorter = mlp(self.seq_len)
        self.sorter.cuda()

        self.criterion_mse = torch.nn.MSELoss()
        self.criterionl1 = torch.nn.L1Loss()

        self.lbd = lbd

    # mem_pred, mem_gt, batch_size, pr=False
    def forward(self, mem_pred, mem_gt, weights=None, pr=False):
        # if batch_size != self.seq_len:
        #     self.sorter = mlp(batch_size)
        #     self.sorter.cuda()
        rank_gt = get_tiedrank(mem_gt)
        rank_pred = self.sorter(mem_pred.unsqueeze(0)).view(-1).cuda()
        # return self.criterion_mse(rank_pred, rank_gt) + self.lbd * self.criterionl1(mem_pred, mem_gt)
        return self.criterion_mse(rank_pred, rank_gt) + self.lbd * weighted_l1_loss(mem_pred, mem_gt, weights)