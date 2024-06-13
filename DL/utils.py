import argparse
import random
import re
from multiprocessing import Pool
from collections import UserList, defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import ast
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
from tdc import Evaluator
from transformers import RobertaTokenizerFast
import torch.nn.functional as F

eps = 1e-8
mse_evaluator = Evaluator(name='MSE')
R2_evaluator = Evaluator(name='R2')
pcc_evaluator = Evaluator(name='PCC')
scc_evaluator = Evaluator(name='Spearman')
rmse_evaluator = Evaluator(name='RMSE')
mae_evaluator = Evaluator(name='MAE')


def get_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return torch.tensor(np.array(fingerprint))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'
    cls = '<cls>'
    mask = '<mask>'


def smiles_tokenizer(smile):
    "Tokenizes SMILES string"
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

def norm(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(mol)
    if Chem.MolFromSmiles(smi):
        return smi
    else:
        return smiles

def hugf_tokenizer(tokenizer, smi):
    tokens = tokenizer(smi, return_tensors='pt', padding=True, max_length=250, truncation=True)
    return tokens['input_ids'][0]


def sequence_tokenizer(data, use_HELM=False):
    def extract_HELM_pattern(data):
        pattern = r"(\d+):R(\d+)-(\d+):R(\d+)"
        match = re.match(pattern, data)
        if match:
            return [match.group(1), "R" + match.group(2), "-", match.group(3), "R" + match.group(4)]
        else:
            return []

    if isinstance(data, tuple):
        seq, HELM = data
    else:
        seq = data
    seq_list = ast.literal_eval(seq)

    if use_HELM:
        HELM_list = extract_HELM_pattern(HELM)
        seq_list.append('<cls>')
        res_list = seq_list + HELM_list
        return res_list

    return seq_list


def get_tokens(datas, config):
    all_tokens = set()
    for data in datas:
        if config['data_type'] == "SMILES":
            token = set(smiles_tokenizer(data))
        elif config['data_type'] == "Sequence":
            token = set(sequence_tokenizer(data, config))
        else:
            print("data type error")
        all_tokens.update(token)
    return all_tokens


def load_token(path):
    with open(path, 'rb') as f:
        loaded_vocab = pickle.load(f)
    return loaded_vocab


def save_token(path, vocab):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)


class WordVocab:
    @classmethod
    def from_data(cls, data, config, *args, **kwargs):
        chars = get_tokens(data, config)
        if config['load_vocab']:
            old_chars = load_token(config['vocab_path'])
            chars = chars | old_chars
            save_token(config['vocab_path'], chars)
        return cls(chars, *args, **kwargs)

    def __init__(self, chars, ss=SpecialTokens):
        if (ss.bos in chars) or (ss.eos in chars) or \
                (ss.pad in chars) or (ss.unk in chars):
            raise ValueError('SpecialTokens in chars')

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk, ss.cls, ss.mask]

        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    @property
    def cls(self):
        return self.c2i[self.ss.cls]

    @property
    def mask(self):
        return self.c2i[self.ss.mask]


def get_vocabulary(data, config):
    return WordVocab.from_data(data, config)


def extent_mask(pre_text_mask, image_feature):
    batch, text_len = pre_text_mask.size()
    _, image_len, _ = image_feature.size()
    if image_feature.is_cuda:
        image_mask = torch.ones(batch, image_len).cuda()
    else:
        image_mask = torch.ones(batch, image_len)

    new_mask = torch.cat((image_mask, pre_text_mask), dim=1)

    return new_mask


def kfold_split(k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    return kf


def initialize_model_parameters(model, skip_names=['image_model']):
    """
    初始化模型中的参数，可以选择跳过指定名称的参数
    Args:
        model: 要初始化的模型
        init_func: 用于初始化参数的初始化函数
        skip_names: 跳过初始化的参数名称列表
    """

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'ln' in name or 'image_model' in name:
                continue
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    # for name, module in model.named_children():
    #     print('1',name)
    #     if name not in skip_names:
    #         for param_name, param in module.named_parameters():
    #             if param.requires_grad and 'ln' not in param_name :
    #                 if 'weight' in param_name:
    #                     print('2',param_name)
    #                     nn.init.xavier_normal_(param)
    #                 elif 'bias' in param_name:
    #                     nn.init.zeros_(param)


import matplotlib.pyplot as plt


def plot_true_vs_predicted(y_true, y_pred):
    """
    绘制真实值与预测值的散点图。

    参数:
    y_true (list): 真实值列表。
    y_pred (list): 预测值列表。
    """
    plt.scatter(y_true, y_pred)  # 绘制散点图
    plt.plot(y_true, y_true, color='red')  # 绘制对角线

    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.savefig('res.png')


def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return R2_evaluator(y_true, y_pred)


def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # plot_true_vs_predicted(y_true,y_pred)
    return mse_evaluator(y_true, y_pred)


def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return rmse_evaluator(y_true, y_pred)


def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return mae_evaluator(y_true, y_pred)


def pearson_correlation_coefficient(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return pcc_evaluator(y_true, y_pred)


def spearman_correlation_coefficient(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return scc_evaluator(y_true, y_pred)


def define_optimizer(model, text_model_config, image_model_config, use_text=True, use_image=True):
    text_optimizer = optim.AdamW(model.text_model.parameters(), lr=text_model_config['lr']) if use_text else None

    image_optimizer = optim.AdamW(model.image_model.parameters(), lr=image_model_config['lr']) if use_image else None

    other_modules = [module for name, module in model.named_children() if name not in ['text_model', 'image_model']]
    other_params = [p for module in other_modules for p in module.parameters()]
    other_optimizer = optim.AdamW(other_params, lr=0.0001) if len(other_params) > 0 else None
    return text_optimizer, image_optimizer, other_optimizer


class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates
        else torch.tensor(-1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=0.3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, weight=None, size_average=True):
        target = torch.zeros((output1.size(0)), output1.size(0)).to(output1.device)
        target.fill_diagonal_(1)

        # 计算 output1 和 output2 的平方差  batch,dim
        # 扩展 output1 和 output2  以便于广播
        output1 = F.normalize(output1, dim=-1)
        output2 = F.normalize(output2, dim=-1)

        modality1_expanded = output1.unsqueeze(1).expand(-1, output2.size(0), -1)
        modality2_expanded = output2.unsqueeze(0).expand(output1.size(0), -1, -1)
        distances_squared = torch.pow(modality1_expanded - modality2_expanded, 2)

        # 求和得到欧氏距离的平方，并开方得到欧氏距离
        euclidean_distances = torch.sqrt(torch.sum(distances_squared, dim=2))

        # 应用边界
        losses = target * euclidean_distances + \
                 (1 + -1 * target).float() * weight * F.relu(self.margin - euclidean_distances)
        return losses.mean() if size_average else losses.sum()


def euclidean_distance_matrix(x):
    """Efficient computation of Euclidean distance matrix
    Args:
      x: Input tensor of shape (batch_size, embedding_dim)

    Returns:
      Distance matrix of shape (batch_size, batch_size)
    """

    # step 1 - compute the dot product
    # shape: (batch_size, batch_size)
    dot_product = torch.mm(x, x.t())
    # step 2 - extract the squared Euclidean norm from the diagonal
    # shape: (batch_size,)
    squared_norm = torch.diag(dot_product)
    # step 3 - compute squared Euclidean distances
    # shape: (batch_size, batch_size)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
    # get rid of negative distances due to numerical instabilities
    distance_matrix = F.relu(distance_matrix)
    # # step 4 - compute the non-squared distances
    # # handle numerical stability
    # # derivative of the square root operation applied to 0 is infinite
    # # we need to handle by setting any 0 to eps

    mask = (distance_matrix == 0.0).float()
    # # # use this mask to set indices with a value of 0 to eps
    distance_matrix = distance_matrix + mask * eps
    # now it is safe to get the square root
    distance_matrix = torch.sqrt(distance_matrix)
    # # undo the trick for numerical stability
    distance_matrix = distance_matrix * (1.0 - mask)

    # distance_matrix = torch.sqrt(distance_matrix + eps)

    return distance_matrix


def get_triplet_mask(labels):
    """compute a mask for valid triplets
    Args:
      labels: Batch of integer labels. shape: (batch_size,)
    Returns:
      Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
      A triplet is valid if:
      `labels[i] == labels[j] and labels[i] != labels[k]`
      and `i`, `j`, `k` are different.
    """
    # step 1 - get a mask for distinct indices
    # shape: (batch_size, batch_size)
    indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)
    # shape: (batch_size, batch_size, 1)
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    # shape: (1, batch_size, batch_size)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    # Shape: (batch_size, batch_size, batch_size)
    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    # step 2 - get a mask for valid anchor-positive-negative triplets
    # shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    # shape: (batch_size, batch_size, 1)
    i_equal_j = labels_equal.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    i_equal_k = labels_equal.unsqueeze(1)
    # shape: (batch_size, batch_size, batch_size)
    valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))
    # step 3 - combine two masks
    mask = torch.logical_and(distinct_indices, valid_indices)
    return mask


class BatchAllTtripletLoss(nn.Module):
    """Uses all valid triplets to compute Triplet loss
    Args:
      margin: Margin value in the Triplet Loss equation
    """

    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """computes loss value.
        Args:
          embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
          labels: Batch of integer labels associated with embeddings. shape: (batch_size,)
        Returns:
          Scalar loss value.
        """
        # step 1 - get distance matrix
        # shape: (batch_size, batch_size)
        distance_matrix = euclidean_distance_matrix(embeddings)
        # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix
        # shape: (batch_size, batch_size, 1)
        anchor_positive_dists = distance_matrix.unsqueeze(2)
        # shape: (batch_size, 1, batch_size)
        anchor_negative_dists = distance_matrix.unsqueeze(1)
        # get loss values for all possible n^3 triplets
        # shape: (batch_size, batch_size, batch_size)
        triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin
        # step 3 - filter out invalid or easy triplets by setting their loss values to 0
        # shape: (batch_size, batch_size, batch_size)
        mask = get_triplet_mask(labels)
        triplet_loss *= mask
        # easy triplets have negative loss values
        triplet_loss = F.relu(triplet_loss)

        # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (triplet_loss > eps).float().sum()
        triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)

        return triplet_loss


class BatchAllTtripletLoss_multi_module_version(nn.Module):
    """Uses all valid triplets to compute Triplet loss
       Args:
         margin: Margin value in the Triplet Loss equation
       """

    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, weight=None):
        """computes loss value.
            output1: text output batch x embedding_dim
            output2: image output batch x embedding_dim
            weight: 1-tanimoto similarity between smiles batch x batch

            if weight == None: plain version
            if weight != None: weight punished version
        """
        output1 = F.normalize(output1, dim=-1)
        output2 = F.normalize(output2, dim=-1)
        cur_batch_size = output1.size(0)
        # step 0 - construct a big vector
        embeddings = torch.cat((output1, output2), dim=0)  # 2 * batch x embedding_dim
        # construct labels for the big vector 0
        labels = torch.cat((torch.arange(cur_batch_size), torch.arange(cur_batch_size)), dim=0).to(
            output1.device)  # 2 * batch

        # step 1 - get distance matrix
        # shape: (2*batch_size, 2*batch_size)
        distance_matrix = euclidean_distance_matrix(embeddings)
        # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix
        # shape: (2*batch_size, 2*batch_size, 1)
        anchor_positive_dists = distance_matrix.unsqueeze(2)
        # shape: (2*batch_size, 1, 2*batch_size)
        anchor_negative_dists = distance_matrix.unsqueeze(1)
        # get loss values for all possible n^3 triplets
        # shape: (2*batch_size, 2*batch_size, 2*batch_size)
        triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin
        # step 3 - filter out invalid or easy triplets by setting their loss values to 0
        # shape: (2*batch_size, 2*batch_size, 2*batch_size)
        mask = get_triplet_mask(labels)
        triplet_loss *= mask
        # # easy triplets have negative loss values
        triplet_loss = F.relu(triplet_loss)
        #
        # # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (triplet_loss > eps).float().sum()
        if weight is not None:
            # if weight == None: plain version
            # if weight != None: weight punished version
            punished_matrix_base = torch.zeros(2 * cur_batch_size, 2 * cur_batch_size).to(output1.device)

            # # Fill the four quadrants
            punished_matrix_base[:cur_batch_size, :cur_batch_size] = weight  # Top-left
            punished_matrix_base[cur_batch_size:, :cur_batch_size] = weight  # Bottom-left
            punished_matrix_base[:cur_batch_size, cur_batch_size:] = weight  # Top-right
            punished_matrix_base[cur_batch_size:, cur_batch_size:] = weight  # Bottom-right
            punished_matrix = punished_matrix_base.unsqueeze(1)  # 2*batch x 1 x 2*batch
            triplet_loss = triplet_loss * punished_matrix

            num_positive_losses = (triplet_loss > eps).float().sum()

        triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)

        return triplet_loss


def draw_feature_distance(distance, similarity):
    import seaborn as sns
    import pandas as pd
    df = pd.DataFrame()
    df['feature_cos_similarity'] = distance.numpy()
    df['tarnimoto_similarity'] = similarity.numpy()
    # 保存 DataFrame 到 CSV 文件
    plot_true_vs_predicted(similarity,distance)
    df.to_csv("all.csv", index=False)

    # df = pd.DataFrame(all_feature.numpy())
    # df.to_csv("0_feature.csv", index=False)
    # print(pearson_correlation_coefficient(distance,similarity))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.kdeplot(distance, shade=True, color='blue', label='Feature Similarity')
    plt.legend()
    plt.subplot(1, 2, 2)
    sns.kdeplot(similarity, shade=True, color='orange', label='Tarnimoto Similarity')
    plt.legend()
    plt.savefig('distance.png')


def feature_similarity(feature1, feature2):
    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(feature1, feature2, dim=1)
    return cosine_similarity


def analyze_feature(img_feature, text_feature, similarity):
    img_feature = torch.tensor(img_feature[0])
    text_feature = torch.tensor(text_feature[0])
    similarity = torch.tensor(similarity[0])

    # 初始化一个 (32, 32) 的相似性矩阵
    feature_similarity_matrix = torch.zeros((64, 64))
    for i in range(img_feature.size(0)):
        for j in range(text_feature.size(0)):
            res = F.cosine_similarity(img_feature[i].unsqueeze(0), text_feature[j].unsqueeze(0))
            res = (res+1)/2
            feature_similarity_matrix[i, j] = res

    draw_feature_distance(feature_similarity_matrix.view(-1) ,similarity.view(-1) )
    
