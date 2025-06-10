from typing import List, Tuple
from .fedavg import FedAvg
import numpy as np
import torch
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import warnings
from utils.finch import FINCH

'''
Hyperparameters (From paper):
    - dataset: random sample from the following domains (digits: 1%, office caltech: 20%)
        - Digits (MNIST, USPS, SVHN, SYN)
        - Office Caltech (Caltech, Amazon, Webcam, DSLR) | 10 overlapping classes between Office31 and Caltech-256
    - model: ResNet-10, feature_dim: 512
    - optimizer: SGD
    - lr: 0.01, momentum: 0.9, weight_decay: 1e-5
    - batch: 64
    - local_epoch: 10
    - communication_rounds: 100
    - num_clients:
        - Digits: 20 (MNIST:3, USPS: 7, SVHN: 6, SYN: 4)
        - Office Caltech: 10 (Caltech: 3, Amazon: 2, Webcam: 1, DSLR: 4)
    - temperature: 0.02
    - aggregation: weighted averaging
'''

class FPL(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.global_protos = {}
        self.local_protos = {}
        self.infoNCET = params["infoNCET"]


    def _initialization(self, **kwargs) -> None:
        pass


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train(global_protos=self.global_protos)
        train_loss, train_acc, ce_loss, proto_loss = result["train_loss"], result["train_acc"], result["ce_loss"], result["proto_loss"]
        self.local_protos[client_list[cid].cid] = client_list[cid].strategy.local_protos
        print("[Client {}] loss: {:.4f}, ce_loss: {:.4f}, proto_loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, ce_loss, proto_loss, train_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)

        self.global_protos = self.aggregate_protos(self.local_protos)
        self.local_protos = {}                                     # clear local_protos
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
        global_protos = kwargs["global_protos"]

        if len(global_protos) != 0:
            all_global_protos_keys = np.array(list(global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())
                mean_f.append(torch.mean(temp_f, dim=0).cpu())
            all_f = [item.detach() for item in all_f]
            mean_f = [item.detach() for item in mean_f]

        model.train()
        for i in range(num_epochs):
            total_loss, total_correct = [], []
            ce_loss, proto_loss = [], []
            for x, labels in trainLoader:
                x, labels = x.to(self.device), labels.to(self.device)
                output, protos = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == labels).sum().item() / len(predict))

                # loss1: cross-entropy loss
                lossCE = F.cross_entropy(output, labels)
                
                # loss2: prototype distance loss
                if len(global_protos) == 0:
                    loss_InfoNCE = 0 * lossCE
                else:
                    loss_InfoNCE = []
                    for idx, label in enumerate(labels):
                        if label.item() in global_protos.keys():
                            f_now = protos[idx].unsqueeze(0)
                            loss_instance = self.hierarchical_info_loss(f_now, label, all_f, mean_f, all_global_protos_keys)    
                            loss_InfoNCE.append(loss_instance)

                    if len(loss_InfoNCE) == 0:
                        loss_InfoNCE = 0 * lossCE
                    else:
                        loss_InfoNCE = torch.mean(torch.stack(loss_InfoNCE, dim=0))
                
                loss = lossCE + loss_InfoNCE
                ce_loss.append(lossCE.item())
                proto_loss.append(loss_InfoNCE.item())
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()

        # generate local ptototypes
        model.eval()
        agg_protos_label = {}
        with torch.no_grad():
            for x, labels in trainLoader:
                x, labels = x.to(self.device), labels.to(self.device)
                _, protos = model(x)
                for k in range(len(labels)):
                    if labels[k].item() in agg_protos_label:
                        agg_protos_label[labels[k].item()].append(protos[k,:])
                    else:
                        agg_protos_label[labels[k].item()] = [protos[k,:]]

        self.local_protos = self.agg_func(agg_protos_label)
        # return np.mean(total_loss), np.mean(total_correct), np.mean(ce_loss), np.mean(proto_loss)
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct),
            "ce_loss": np.mean(ce_loss),
            "proto_loss": np.mean(proto_loss)
        }


    def _test(self, model, testLoader, **kwargs) -> Tuple[float, float]:
        ''' Test function for the client '''
        total_loss, total_correct = [], []
        num_classes = self.args.num_classes
        correct_predictions = {i: 0 for i in range(num_classes)}
        total_counts = {i: 0 for i in range(num_classes)}

        model.eval()
        with torch.no_grad():
            for x, label in testLoader:
                x, label = x.to(self.device), label.to(self.device)
                output, _ = model(x)
                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))
                total_loss.append(F.cross_entropy(output, label).item())

                for i in range(num_classes):
                    mask = label == i
                    correct_predictions[i] += (predict[mask] == label[mask]).sum().item()
                    total_counts[i] += mask.sum().item()

        class_acc = {i: (correct_predictions[i] / total_counts[i]) if total_counts[i] > 0 else 0 for i in range(num_classes)}
        # return np.mean(total_loss), np.mean(total_correct), class_acc
        return {
            "test_loss": np.mean(total_loss),
            "test_acc": np.mean(total_correct),
            "class_acc": class_acc
        }


    def agg_func(self, protos):
        """
        Returns the average of the weights.
        """
        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]

        return protos
    
    
    def aggregate_protos(self, local_protos_list):
        agg_protos_label = dict()
        for idx in local_protos_list:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]
        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)

                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                    agg_selected_proto.append(torch.tensor(proto))
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = [proto_list[0].unsqueeze(0)]

        return agg_protos_label


    def hierarchical_info_loss(self, f_now, label, all_f, mean_f, all_global_protos_keys):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # print(label, all_global_protos_keys)
            # for i in all_f:
            #     print(i, i.shape)

            f_pos = np.array(all_f, dtype=object)[all_global_protos_keys == label.item()][0].to(self.device)
            tmp_neg = np.array(all_f, dtype=object)[all_global_protos_keys != label.item()]
            if len(tmp_neg) == 0:
                f_neg = f_now
            else:
                f_neg = torch.cat(list(np.array(all_f, dtype=object)[all_global_protos_keys != label.item()])).to(self.device)
            xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)

            mean_f_pos = np.array(mean_f, dtype=object)[all_global_protos_keys == label.item()][0].to(self.device)
            mean_f_pos = mean_f_pos.view(1, -1)
            # mean_f_neg = torch.cat(list(np.array(mean_f)[all_global_protos_keys != label.item()]), dim=0).to(self.device)
            # mean_f_neg = mean_f_neg.view(9, -1)

            cu_info_loss = F.mse_loss(f_now, mean_f_pos).to(self.device)

            hierar_info_loss = xi_info_loss + cu_info_loss
        return hierar_info_loss


    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1).to(self.device)
        l = l / self.infoNCET

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float, device=self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss