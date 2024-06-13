import os

from model import BiRNNRegression, Transformer, TextCNN
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_vocabulary, r2_score, mse, mae, pearson_correlation_coefficient, spearman_correlation_coefficient
import numpy as np
from tqdm import tqdm
import yaml
from contextlib import nullcontext


def train(model, train_loader, test_loader, val_loader, text_optimizer, image_optimizer, other_optimizer, criterion,
          device, epochs, writer, log_file):
    for epoch in range(epochs):
        if epoch % 10 == 0:  # Every 10 epochs
            writer.flush()
        model.train()
        loss = 0
        for batch_idx, (data, target, length, image, fg, weight,_) in enumerate(
                tqdm(train_loader, desc="Training batches")):
            data = data.to(device)
            target = target.to(device)
            image = image.to(device) if image[0] is not None else image
            fg = fg.to(device)
            weight = weight.to(device)
            if epoch <= 10 and text_optimizer is not None:
                text_optimizer.zero_grad()
                if other_optimizer is not None:
                    other_optimizer.zero_grad()

                output, CL_loss = model(data, image, length, fg, weight)
                target = target.view(-1, 1)
                train_loss = criterion(output, target)
                if image_optimizer is not None:
                    train_loss += CL_loss
                loss += train_loss.item()

                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                text_optimizer.step()
                if other_optimizer is not None:
                    other_optimizer.step()

            else:
                if text_optimizer is not None:
                    text_optimizer.zero_grad()
                if image_optimizer is not None:
                    image_optimizer.zero_grad()
                if other_optimizer is not None:
                    other_optimizer.zero_grad()

                output, CL_loss = model(data, image, length, fg, weight)
                target = target.view(-1, 1)
                train_loss = criterion(output, target)
                loss += train_loss.item()

                if image_optimizer is not None:
                    train_loss += CL_loss

                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)

                if text_optimizer is not None:
                    text_optimizer.step()
                if image_optimizer is not None:
                    image_optimizer.step()
                if other_optimizer is not None:
                    other_optimizer.step()

            if hasattr(model, "use_ema") and model.use_ema:
                model.model_ema(model)

        loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.4f}")
        writer.add_scalar('training loss', loss, epoch + 1)

        # Test
        model.eval()


        val_loss = 0
        all_y_true = []
        all_y_pred = []
        with context_manager and torch.no_grad():
            for batch_idx, (data, target, length, image, fg, weight,_) in enumerate(tqdm(val_loader, desc="Val batches")):
                data = data.to(device)
                target = target.to(device)
                image = image.to(device) if image[0] is not None else image
                fg = fg.to(device)
                weight = weight.to(device)

                output, CL_loss = model(data, image, length, fg, weight)
                output = output.cpu().flatten()

                target = target.view(-1, 1).cpu().flatten()

                # R2
                all_y_true.append(target)
                all_y_pred.append(output)
        val_R2_score = r2_score(torch.cat(all_y_true), torch.cat(all_y_pred))
        val_loss = mse(torch.cat(all_y_true), torch.cat(all_y_pred))
        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, R2_score: {val_R2_score:.4f}")

        writer.add_scalar('val loss', val_loss, epoch + 1)
        writer.add_scalar('val R2_score', val_R2_score, epoch + 1)
        
        test_loss = 0
        all_y_true = []
        all_y_pred = []
        context_manager = model.ema_scope(context="ema version") if hasattr(model,
                                                                            'use_ema') and model.use_ema else nullcontext
        with context_manager and torch.no_grad():
            for batch_idx, (data, target, length, image, fg, weight,_) in enumerate(
                    tqdm(test_loader, desc="Testing batches")):
                data = data.to(device)
                # target = target.to(device)
                image = image.to(device) if image[0] is not None else image
                fg = fg.to(device)
                weight = weight.to(device)
                output, CL_loss = model(data, image, length, fg, weight)
                output = output.cpu().flatten()

                target = target.view(-1, 1).cpu().flatten()

                # R2
                all_y_true.append(target)
                all_y_pred.append(output)

        test_mae = mae(torch.cat(all_y_true), torch.cat(all_y_pred))
        test_R2_score = r2_score(torch.cat(all_y_true), torch.cat(all_y_pred))
        test_pearsonr = pearson_correlation_coefficient(torch.cat(all_y_true), torch.cat(all_y_pred))
        test_spearmanr = spearman_correlation_coefficient(torch.cat(all_y_true), torch.cat(all_y_pred))
        test_loss = mse(torch.cat(all_y_true), torch.cat(all_y_pred))
        test_criterion = test_loss + 1 - test_R2_score + 1 - test_pearsonr

        print(f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}, R2_score: {test_R2_score:.4f}")
        print(f"Test MAE {test_mae:.4f}, Test PCC: {test_pearsonr:.4f}, SCC: {test_spearmanr:.4f}")
        writer.add_scalar('test loss', test_loss, epoch + 1)
        writer.add_scalar('test R2_score', test_R2_score, epoch + 1)
        writer.add_scalar('test mae', test_mae, epoch + 1)
        writer.add_scalar('test pcc', test_pearsonr, epoch + 1)
        writer.add_scalar('test scc', test_spearmanr, epoch + 1)

        
