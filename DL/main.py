import argparse

from transformers import RobertaTokenizerFast, AutoTokenizer

from dataset import SMILESDataset, collate_wrapper
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from model import Peptide_Regression
import torch
import torch.nn as nn
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)
from utils import get_vocabulary, str2bool, initialize_model_parameters, define_optimizer
import numpy as np
from train import train
import datetime
from torch.utils.tensorboard import SummaryWriter

import yaml
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def set_log(config):
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('../log/runs/{}'.format(current_time))

    filename = "../log/runs/{}.txt".format(current_time)
    with open(filename, "w") as f:
        f.write("Config:\n")
        f.write(config['data_yaml']['text_data_yaml'])
        f.write(yaml.dump(config))

    return writer, filename


def add_log(filename, config):
    with open(filename, "a") as f:
        f.write("\n")
        f.write(yaml.dump(config))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Peptide Regression')
    parser.add_argument('--all_config', type=str, default='config/model.yaml',
                        help='Path to the config file.')

    parser.add_argument("--text_data_yaml", type=str, default=None),
    parser.add_argument("--image_data_yaml", type=str, default=None),

    parser.add_argument("--text_model_yaml", type=str, default=None),
    parser.add_argument("--image_model_yaml", type=str, default=None),

    parser.add_argument("--use_text_info", type=str2bool, default=True),
    parser.add_argument("--use_image_info", type=str2bool, default=True),
    parser.add_argument("--feature_cmb_type", type=str, default='concate'),
    # 可以根据需要添加更多的参数
    return parser.parse_args()


def main():
    args = parse_arguments()
    all_config = load_config(args.all_config)

    # 用args覆盖config
    all_config['model_yaml']['use_text_info'] = args.use_text_info
    all_config['model_yaml']['use_image_info'] = args.use_image_info
    if args.text_data_yaml is not None:
        all_config['data_yaml']['text_data_yaml'] = args.text_data_yaml
    if args.image_data_yaml is not None:
        all_config['data_yaml']['image_data_yaml'] = args.image_data_yaml
    if args.text_model_yaml is not None:
        all_config['model_yaml']['text_model_yaml'] = args.text_model_yaml
    if args.image_model_yaml is not None:
        all_config['model_yaml']['image_model_yaml'] = args.image_model_yaml
    if args.feature_cmb_type is not None:
        all_config['model_yaml']['feature_cmb_type'] = args.feature_cmb_type

    writer, log_file = set_log(all_config)

    data_yaml_config = all_config['data_yaml']
    model_yaml_config = all_config['model_yaml']

    text_data_yaml = data_yaml_config['text_data_yaml']
    image_data_yaml = data_yaml_config['image_data_yaml']

    text_model_yaml = model_yaml_config['text_model_yaml']
    image_model_yaml = model_yaml_config['image_model_yaml']

    # 读取数据的yaml内容
    text_data_config = load_config(text_data_yaml)
    text_data_type = text_data_config['data_type']

    image_data_config = load_config(image_data_yaml)

    target_type = text_data_config['target']  # 要预测的回归value的类型

    # 读取模型的yaml内容
    text_model_config = load_config(text_model_yaml)
    text_model_type = text_model_config['model_type']
    batch_size = text_model_config['batch_size']

    image_model_config = load_config(image_model_yaml)

    # 读取text数据，目前image根据text的id去Data类匹配
    image_data_folder = image_data_config['image_folder'] if args.use_image_info else None
    # 训练集
    text_train_df = pd.read_csv(text_data_config['train_data_path'])
    train_text_data_list = text_train_df[text_data_type].values
    train_targets = text_train_df[target_type].values
    train_cyclepeptideID = text_train_df['CycPeptMPDB_ID'].values

    # 测试集
    text_test_df = pd.read_csv(text_data_config['test_data_path'])
    test_text_data_list = text_test_df[text_data_type].values
    test_targets = text_test_df[target_type].values
    test_cyclepeptideID = text_test_df['CycPeptMPDB_ID'].values

    # 验证集
    text_val_df = pd.read_csv(text_data_config['val_data_path'])
    val_text_data_list = text_val_df[text_data_type].values
    val_targets = text_val_df[target_type].values
    val_cyclepeptideID = text_val_df['CycPeptMPDB_ID'].values

    add_log(log_file, text_data_config)

    # 建立vocab

    if text_model_type == 'PubChemLM':
        vocab = AutoTokenizer.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_150k")
    elif text_model_type == 'ChemBERTa':
        vocab = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    else:
         vocab = get_vocabulary(list(train_text_data_list) + list(test_text_data_list) + list(val_text_data_list),
                                   text_data_config)
    train_dataset = SMILESDataset(train_text_data_list,
                                      train_targets,
                                      vocab,
                                      train_cyclepeptideID,
                                      image_data_folder,
                                      text_data_config,
                                      image_size=image_model_config['image_size'],
                                      image_augment=image_data_config['augmentation'],
                                      smiles_augment=True)
    test_dataset = SMILESDataset(test_text_data_list,
                                     test_targets,
                                     vocab,
                                     test_cyclepeptideID,
                                     image_data_folder,
                                     text_data_config,
                                     image_size=image_model_config['image_size'],
                                     image_augment=False,
                                     smiles_augment=False)
    val_dataset = SMILESDataset(val_text_data_list,
                                    val_targets,
                                    vocab,
                                    val_cyclepeptideID,
                                    image_data_folder,
                                    text_data_config,
                                    image_size=image_model_config['image_size'],
                                    image_augment=False,
                                    smiles_augment=False)

    # 建立dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper(vocab),
                              drop_last=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_wrapper(vocab))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_wrapper(vocab))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Peptide_Regression(all_model_config=model_yaml_config,
                               text_vocab=vocab,
                               text_model_config=text_model_config,
                               text_data_config=text_data_config,
                               image_model_config=image_model_config,
                               image_data_config=image_data_config)
    with open(log_file, "a") as f:
        f.write('Network structure:\n')
        f.write(str(model))
        f.write('\n\n')

    # initialize_model_parameters(model)
    model.to(device)
    MSE_criterion = nn.MSELoss()

    text_optimizer, image_optimizer, other_optimizer = define_optimizer(model, text_model_config, image_model_config,
                                                                        use_text=model_yaml_config['use_text_info'],
                                                                        use_image=model_yaml_config['use_image_info'])
    # optimizer = optim.AdamW(model.parameters(), lr=text_model_config['lr'])

    """# sche control learning stage
    # adjust learning rate
    # train text for the first 20 epoch and then train the two model together
    # 1  for text
    optimizer_text = optim.AdamW(model_text.parameters(), lr=text_model_config['lr'])
    # 2  for image
    optimizer_image = optim.AdamW(model_image.parameters(), lr=text_model_config['lr'])"""

    train(model=model, train_loader=train_loader, test_loader=test_loader, val_loader=val_loader,
          text_optimizer=text_optimizer, image_optimizer=image_optimizer,
          other_optimizer=other_optimizer, criterion=MSE_criterion, device=device, epochs=text_model_config['epochs'],
          writer=writer, log_file=log_file)
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
