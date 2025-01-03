import argparse
import os

from transformers import RobertaTokenizerFast, AutoTokenizer

from DL.dataset import SMILESDataset, collate_wrapper
from torch.utils.data import Dataset, DataLoader, random_split

from DL.draw_attention import draw_SHAP
from model import Peptide_Regression
import torch
import pandas as pd
from utils import get_vocabulary, r2_score, mse, mae, pearson_correlation_coefficient, spearman_correlation_coefficient, \
    rmse, analyze_feature
import numpy as np
from tqdm import tqdm
import yaml
from contextlib import nullcontext


def test(model, test_loader, device):
    model.eval()

    all_y_true = []
    all_y_pred = []

    all_text_feature = []
    all_img_feature = []
    all_feature_simi = []

    context_manager = model.ema_scope(context="ema version") if hasattr(model,
                                                                        'use_ema') and model.use_ema else nullcontext
    with context_manager and torch.no_grad():
        for batch_idx, (data, target, length, image, fg, weight, cur_cyclepeptideID) in enumerate(
                tqdm(test_loader, desc="Testing batches")):
            data = data.to(device)
            # target = target.to(device)
            image = image.to(device) if image[0] is not None else image
            fg = fg.to(device)
            weight = weight.to(device)

            ### 不需要返回特征
            # output, _ = model(data, image, length, fg, weight=None, return_feature=True)
            ###

            output, _, text_feature, image_feature = model(data, image, length, fg, weight=None, return_feature=True)
            all_text_feature.append(text_feature.cpu())
            all_img_feature.append(image_feature.cpu())
            all_feature_simi.append(1 - weight.cpu())

            output = output.cpu().flatten()

            target = target.view(-1, 1).cpu().flatten()

            # R2
            all_y_true.append(target)
            all_y_pred.append(output)

    # analyze_feature(all_img_feature, all_text_feature, all_feature_simi)

    all_y_true = torch.cat(all_y_true)
    all_y_pred = torch.cat(all_y_pred)
    test_R2_score = r2_score(all_y_true, all_y_pred)
    test_mse = mse(all_y_true, all_y_pred)
    test_mae = mae(all_y_true, all_y_pred)
    test_rmse = rmse(all_y_true, all_y_pred)
    test_pcc = pearson_correlation_coefficient(all_y_true, all_y_pred)
    test_scc = spearman_correlation_coefficient(all_y_true, all_y_pred)
    print(f"Test_MSE: {test_mse:.4f}, "
          f"Test_RMSE: {test_rmse:.4f}, "
          f"Test_MAE: {test_mae:.4f}, "
          f"R2_score: {test_R2_score:.4f} "
          f"Test_pcc: {test_pcc:.4f}, "
          f"Test_scc: {test_scc:.4f}, "
          )
    return np.array(all_y_pred)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Peptide Regression')
    parser.add_argument('--all_config', type=str, default='config/model.yaml',
                        help='Path to the config file.')
    parser.add_argument('--data_path', type=str, default='../data/test.csv',
                        help='test data path.')
    parser.add_argument('--model_path', type=str,
                        help='model ckpt path.')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab.pkl',
                        help='vocab path.')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    args = parse_arguments()
    test_data_path = args.data_path
    model_ckpt_path = args.model_path
    vocab = args.vocab_path
    all_config = load_config(args.all_config)

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
    image_data_folder = image_data_config['image_folder']

    # 测试集
    text_test_df = pd.read_csv(test_data_path)

    text_test_df = text_test_df[text_test_df['is_cliff']]

    test_text_data_list = text_test_df[text_data_type].values
    test_targets = text_test_df[target_type].values
    test_cyclepeptideID = text_test_df['CycPeptMPDB_ID'].values

    if text_model_type == 'PubChemLM':
        vocab = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    elif text_model_type == 'ChemBERTa':
        vocab = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    else:
        vocab = get_vocabulary(list(test_text_data_list), text_data_config)

    test_dataset = SMILESDataset(test_text_data_list,
                                 test_targets,
                                 vocab,
                                 test_cyclepeptideID,
                                 image_data_folder,
                                 text_data_config,
                                 image_size=image_model_config['image_size'],
                                 image_augment=False,
                                 smiles_augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_wrapper(vocab))

    model = Peptide_Regression(all_model_config=model_yaml_config,
                               text_vocab=vocab,
                               text_model_config=text_model_config,
                               text_data_config=text_data_config,
                               image_model_config=image_model_config,
                               image_data_config=image_data_config)
    model.load_state_dict(torch.load(model_ckpt_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pred = test(model, test_loader=test_loader, device=device)

    # text_test_df['pred_chembert'] = pred
    # text_test_df.to_csv(test_data_path,index=False)
