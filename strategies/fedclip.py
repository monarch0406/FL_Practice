from typing import List, Tuple
from .fedavg import FedAvg
import numpy as np
import torch
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from utils.loss_fn import ContrastiveLoss

class FedCLIP(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.criterion = ContrastiveLoss()


    def _initialization(self, **kwargs) -> None:
        pass


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train()
        train_loss = result["train_loss"]
        print("[Client {}] loss: {:.4f}".format(cid, train_loss))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
        model.train()
        for i in range(num_epochs):
            total_loss, total_correct = [], []
            for images, input_ids in trainLoader:
                images, input_ids = images.to(self.device), input_ids.to(self.device)

                # print("Image batch shape:", images.shape)
                # print("Text batch shape:", input_ids.shape)
                # input()

                image_feat, text_feat = model(images, input_ids)
                
                loss = self.criterion(image_feat, text_feat)
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()

        return {
            "train_loss": np.mean(total_loss),
        }
    

    def _test(self, model, testLoader, **kwargs) -> Tuple[float, float]:
        ''' Test function for the client '''
        model.eval()
        all_image_feats = []
        all_text_feats = []

        with torch.no_grad():
            for images, input_ids in testLoader:
                images, input_ids = images.to(self.device), input_ids.to(self.device)
                image_feat, text_feat = model(images, input_ids)
                all_image_feats.append(image_feat)
                all_text_feats.append(text_feat)

        image_feats = torch.cat(all_image_feats, dim=0)
        text_feats = torch.cat(all_text_feats, dim=0)
        image_feats = torch.nn.functional.normalize(image_feats, dim=1)
        text_feats = torch.nn.functional.normalize(text_feats, dim=1)

        sim = image_feats @ text_feats.t()
        ranks = sim.argsort(dim=1, descending=True)
        targets = torch.arange(sim.size(0)).to(sim.device)

        r1 = (ranks[:, 0] == targets).float().mean().item()
        r5 = (ranks[:, :5] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        r10 = (ranks[:, :10] == targets.unsqueeze(1)).any(dim=1).float().mean().item()

        print(f"Image -> Text Recall@1: {r1:.3f}, Recall@5: {r5:.3f}, Recall@10: {r10:.3f}")
        return {
            "test_loss": 0,
            "test_acc": 0,
            "class_acc": [0 for _ in range(self.args.num_classes)],
        }