import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

from transformers import RobertaTokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerFast

from utils import smiles_tokenizer, sequence_tokenizer, hugf_tokenizer, get_fingerprint
from rdkit import Chem
import os

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rdkit import Chem
import pandas as pd


##############################################################################################################
# downside for image
##############################################################################################################

class CropWhite(A.DualTransform):

    def __init__(self, value=(255, 255, 255), pad=0, p=1.0):
        super(CropWhite, self).__init__(p=p)
        self.value = value
        self.pad = pad
        assert pad >= 0

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        assert "image" in kwargs
        img = kwargs["image"]
        height, width, _ = img.shape
        x = (img != self.value).sum(axis=2)
        if x.sum() == 0:
            return params
        row_sum = x.sum(axis=1)
        top = 0
        while row_sum[top] == 0 and top + 1 < height:
            top += 1
        bottom = height
        while row_sum[bottom - 1] == 0 and bottom - 1 > top:
            bottom -= 1
        col_sum = x.sum(axis=0)
        left = 0
        while col_sum[left] == 0 and left + 1 < width:
            left += 1
        right = width
        while col_sum[right - 1] == 0 and right - 1 > left:
            right -= 1
        # crop_top = max(0, top - self.pad)
        # crop_bottom = max(0, height - bottom - self.pad)
        # crop_left = max(0, left - self.pad)
        # crop_right = max(0, width - right - self.pad)
        # params.update({"crop_top": crop_top, "crop_bottom": crop_bottom,
        #                "crop_left": crop_left, "crop_right": crop_right})
        params.update({"crop_top": top, "crop_bottom": height - bottom,
                       "crop_left": left, "crop_right": width - right})
        return params

    def apply(self, img, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        height, width, _ = img.shape
        img = img[crop_top:height - crop_bottom, crop_left:width - crop_right]
        img = A.augmentations.pad_with_params(
            img, self.pad, self.pad, self.pad, self.pad, border_mode=cv2.BORDER_CONSTANT, value=self.value)
        return img

    def apply_to_keypoint(self, keypoint, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        x, y, angle, scale = keypoint[:4]
        return x - crop_left + self.pad, y - crop_top + self.pad, angle, scale

    def get_transform_init_args_names(self):
        return ('value', 'pad')


class SaltAndPepperNoise(A.DualTransform):

    def __init__(self, num_dots, value=(0, 0, 0), p=0.5):
        super().__init__(p)
        self.num_dots = num_dots
        self.value = value

    def apply(self, img, **params):
        height, width, _ = img.shape
        num_dots = random.randrange(self.num_dots + 1)
        for i in range(num_dots):
            x = random.randrange(height)
            y = random.randrange(width)
            img[x, y] = self.value
        return img

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ('value', 'num_dots')


def get_transforms(input_size, augment=True, debug=False):
    trans_list = []
    if augment:
        trans_list += [
            A.SafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
            CropWhite(pad=5),
            A.CropAndPad(percent=[-0.01, 0.00], keep_size=False, p=0.5),
            A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
            A.Blur(),
            A.GaussNoise(),
            SaltAndPepperNoise(num_dots=20, p=0.5)
        ]
    trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        trans_list += [
            A.Normalize(),
            ToTensorV2(),
        ]
    return A.Compose(trans_list)


##############################################################################################################
# downside for Text
##############################################################################################################

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, targets, vocab, cyclepeptideID, image_folder, text_data_config, image_size,
                 image_augment=False, smiles_augment=False):
        self.smiles_list = smiles_list
        self.targets = targets
        self.vocab = vocab
        self.max_len = text_data_config['max_len']
        self.augment = text_data_config['augmentation'] and smiles_augment
        self.augment_ratio = text_data_config['augmentation_ratio']
        self.cyclepeptideID = cyclepeptideID
        self.image_folder = image_folder
        self.transform = get_transforms(image_size, augment=image_augment)

    def __len__(self):
        return len(self.smiles_list)

    def image_transform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        augmented_image = self.transform(image=image)["image"]

        return augmented_image

    def randomize_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        smi = Chem.MolToSmiles(mol, doRandom=True)
        if Chem.MolFromSmiles(smi):
            return smi
        else:
            return smiles

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        # add fingerprint
        fingerprint = get_fingerprint(smi)

        if self.augment:
            if random.random() > (1 - self.augment_ratio):
                smi = self.randomize_smiles(smi)

        if isinstance(self.vocab, (PreTrainedTokenizer, PreTrainedTokenizerFast)):  # 用huggingface的
            smi_token_ids = hugf_tokenizer(self.vocab, smi)

        else:  # 用自己的分词器
            smi = smiles_tokenizer(smi)
            smi_token_ids = [self.vocab.c2i[char] for char in smi]
            # data_token_ids.append(self.vocab.eos)  # 在末尾添加
            smi_token_ids.insert(0, self.vocab.bos)  # 在开头添加

        actual_length = len(smi_token_ids)  # 记录实际长度

        target = self.targets[idx]
        data_token_ids = torch.tensor(smi_token_ids, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.float32)

        cur_cyclepeptideID = self.cyclepeptideID[idx]
        if self.image_folder is not None:
            filename = "{}.png".format(cur_cyclepeptideID)
            cur_png_path = os.path.join(self.image_folder, filename)
            image = cv2.imread(cur_png_path)
            image = self.image_transform(image)
        else:
            image = None

        return data_token_ids, target, actual_length, image, fingerprint, cur_cyclepeptideID


def get_similarity_matrix():
    similarity_matrix_df = pd.read_pickle('KG.pkl')
    return similarity_matrix_df


similarity_matrix_df = get_similarity_matrix()


def collate_wrapper(vocab):
    def collate_fn(batch):
        # Sort batch by sequence length
        batch = sorted(batch, key=lambda x: x[2], reverse=True)
        sequences, targets, lengths, images, fingerprint, cur_cyclepeptideID = zip(*batch)

        # Pad sequences
        if isinstance(vocab, (PreTrainedTokenizer, PreTrainedTokenizerFast)):  # 用huggingface的
            seqs_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=vocab.pad_token_id)
        else:
            seqs_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=vocab.pad)

        # Convert lengths to tensor
        lengths = torch.tensor(lengths, dtype=torch.long)

        images = torch.stack(images) if images[0] is not None else images
        if isinstance(cur_cyclepeptideID[0], str):
            weight = torch.zeros_like(lengths)
        else:
            weight = 1 - similarity_matrix_df.loc[list(cur_cyclepeptideID), list(cur_cyclepeptideID)]
            weight = torch.tensor(weight.values)

        return seqs_padded, torch.stack(targets), lengths, images, torch.stack(fingerprint), weight, cur_cyclepeptideID

    return collate_fn
