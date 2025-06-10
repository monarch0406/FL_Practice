from typing import List, Tuple
from colorama import Fore, Style
from .fedavg import FedAvg
import numpy as np
import math
import copy
from tqdm import tqdm
import torch
from torch import nn
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from functools import reduce
from torch.utils.data import DataLoader
from models.model import CGenerator
from utils.loss_fn import DiversityLoss
np.set_printoptions(suppress=True, linewidth=np.inf)

'''
Hyperparameters:
    - dataset: MNIST, CIFAR-10, Office-Caltech
'''

class FedDPFL(FedAvg):
    def __init__(self, fl_strategy, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.fl_strategy = fl_strategy
        ''' Parameters for data-free knowledge distillation '''
        self.z_dim = 100
        self.cgan = CGenerator(nz=self.z_dim, ngf=16, img_size=32, n_cls=args.num_classes).to(device)
        self.optimizer_cgan = torch.optim.Adam(self.cgan.parameters(), lr=3e-4, weight_decay=1e-2)
        # self.scheduler_cgan = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_cgan, gamma=0.98)

        ''' Hyperparameters for knowledge distillation'''
        self.batch_size = args.batch_size
        self.gen_batch_size = int(params['gen_batch_size'])
        self.iterations = int(params['iterations'])
        self.inner_round_g = int(params['inner_round_g'])
        self.inner_round_d = int(params['inner_round_d'])
        self.T = params['con_T']

        ''' Coefficients for individual loss '''
        self.ensemble_gamma = params['kd_gamma']
        self.ensemble_beta = params['kd_beta']
        self.ensemble_eta = params['kd_eta']
        self.age_ld = params['age_ld']
        self.impt_ld = params['impt_ld']
        
        ''' Parameters for client data statistics '''
        self.num_classes = args.num_classes
        self.label_weights = []
        self.qualified_labels = []
    
        self.criterion_diversity = DiversityLoss(metric='l1').to(device)
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        self.valLoader = None

        '''Knowledge Pool 
            - {
                signature (data_classes): {
                    "model": model,
                    "data_summary": {
                        "label": num_data
                    },
                    "num_data": len(data),
                    "online_age": rounds,
                    "offline_age": rounds,
                }
            }
        '''
        self.knowledge_pool = {}
        self.online_rate = params['online_rate']
        self.offline_rate = params['offline_rate']


    # def initialization(self,):
    #     self.cgan = CGenerator().to(self.device)
    #     self.optimizer_cgan = torch.optim.Adam(self.cgan.parameters(), lr=0.01)
    #     self.scheduler_cgan = torch.optim.lr_scheduler.StepLR(self.optimizer_cgan, step_size=1, gamma=0.998)


    def data_free_knowledge_distillation(self, global_model, optimizer_server, scheduler_server, rounds):
        generator = self.cgan
        self.label_weights, self.qualified_labels = self.get_label_weights_from_knowledge_pool(self.knowledge_pool)
        age_weights = self.get_age_weight()

        ''' Integrate the label weight and age weight '''
        self.label_weights = self.combine_weights(self.label_weights, age_weights * self.age_ld)
        total_label_weights = np.sum(self.label_weights, axis=0)

        intial_val_acc = self.evaluate(global_model, self.valLoader)      # inital val_acc
        state = {
            "best_val_acc": intial_val_acc,
            "best_server_model": copy.deepcopy(global_model.state_dict()),
            "best_generator": copy.deepcopy(generator.state_dict()),
        }
        intial_test_acc = self.evaluate(global_model, self.testLoader)      # inital test_acc
        print("[Server | E. Distillation] Start ensemble distillation, inital test_acc: {:.4f}".format(intial_test_acc))
        
        pbar = tqdm(range(self.iterations), desc="[Server | E. Knowledge Distillation]", leave=False)
        for _ in pbar:
            ''' Train Generator '''
            generator.train()
            global_model.eval()

            loss_G_total = []
            loss_KD_total = []
            loss_con_total = []
            loss_cls_total = []
            loss_div_total = []

            # y = np.random.choice(self.qualified_labels, self.batch_size)
            y = self.generate_labels(self.batch_size, total_label_weights)

            y_input = F.one_hot(torch.tensor(y), num_classes=self.args.num_classes).type(torch.float32).to(self.device)

            for _ in range(self.inner_round_g):
                ''' feed to generator '''
                z = torch.randn((self.batch_size, self.z_dim), device=self.device)
                gen_output = generator(z, y_input)

                ''' get the student logit '''
                _, student_feature = global_model(gen_output)

                ''' compute diversity loss '''
                loss_div = self.criterion_diversity(z, gen_output)


                ''' Train the generator using contrastive learning to separate the features (class separation) '''
                # create the class feature
                class_features = [[] for _ in range(self.num_classes)]
                for i in range(self.batch_size):
                    class_features[y[i]].append(student_feature[i])
                class_features = [torch.stack(class_feature) if len(class_feature) > 0 else torch.zeros((1, student_feature.shape[1]), device=self.device) for class_feature in class_features]

                # for i in range(self.num_classes):
                #     print("class_features[{}]: {}".format(i, class_features[i].shape))

                ''' create positive pairs and negative pairs of each class '''
                features_pos, features_neg = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
                for i in range(self.num_classes):
                    features_pos[i] = torch.mean(class_features[i], dim=0).view(1, -1)
                    features_neg[i] = torch.stack([torch.mean(class_features[j], dim=0) for j in range(self.num_classes) if j != i])
                    # print("features_pos[{}]: {}, features_neg[{}]: {}".format(i, features_pos[i].shape, i, features_neg[i].shape))
                
                # create the positive and negative pairs for each data
                pos_pairs, neg_pairs = [], []
                for k in range(self.batch_size):
                    pos_pairs.append(features_pos[y[k]])
                    neg_pairs.append(features_neg[y[k]])

                pos_pairs = torch.stack(pos_pairs)
                neg_pairs = torch.stack(neg_pairs)
                # print("pos_pairs: {}, neg_pairs: {}".format(pos_pairs.shape, neg_pairs.shape))

                pos_sim = self.cosine_sim(student_feature.unsqueeze(1), pos_pairs)
                neg_sim = self.cosine_sim(student_feature.unsqueeze(1), neg_pairs)
                # print("pos_sim: {}, neg_sim: {}".format(pos_sim.shape, neg_sim.shape))

                logits = torch.cat([pos_sim, neg_sim], dim=1).to(self.device)
                logits /= self.T
                # print("logits:", logits.shape)

                target = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
                loss_con = F.cross_entropy(logits, target)
 

                loss_cls = 0
                for idx, (sig, value) in enumerate(self.knowledge_pool.items()):
                    _teacher = value["model"]
                    _teacher.eval()

                    weight = self.label_weights[y][:, idx].reshape(-1, 1)
                    weight = torch.tensor(weight.squeeze(), dtype=torch.float32, device=self.device)

                    teacher_logit, _ = _teacher(gen_output)
                    loss_cls += torch.mean(F.cross_entropy(teacher_logit, y_input) * weight)

                loss = self.ensemble_gamma * loss_con + self.ensemble_beta * loss_cls + self.ensemble_eta * loss_div
                loss_con_total.append(loss_con.item() * self.ensemble_gamma)
                loss_cls_total.append(loss_cls.item() * self.ensemble_beta)
                loss_div_total.append(loss_div.item() * self.ensemble_eta)
                loss_G_total.append(loss.item())

                self.optimizer_cgan.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 10)               # clip the gradient to prevent exploding
                self.optimizer_cgan.step()

            # ''' save the images from the generator '''
            gen_output = F.interpolate(gen_output, scale_factor=2, mode='bilinear', align_corners=False)
            save_image(make_grid(gen_output[:8], nrow=8, normalize=True), f"./figures/gen_output.png")

            ''' Train student (global model) '''
            generator.eval()
            global_model.train()

            ''' Sample new data '''
            # y = np.random.choice(self.qualified_labels, self.gen_batch_size)
            y = self.generate_labels(self.gen_batch_size, total_label_weights)
            y_input = F.one_hot(torch.tensor(y), num_classes=self.args.num_classes).type(torch.float32).to(self.device)

            for _ in range(self.inner_round_d):
                z = torch.randn((self.gen_batch_size, self.z_dim), device=self.device)
                gen_output = generator(z, y_input)
                student_logit, _ = global_model(gen_output)

                t_logit_merge = 0
                with torch.no_grad():
                    for idx, (sig, value) in enumerate(self.knowledge_pool.items()):
                        _teacher = value["model"]
                        _teacher.eval()

                        weight = self.label_weights[y][:, idx].reshape(-1, 1)
                        expand_weight = np.tile(weight, (1, self.num_classes))

                        teacher_logit, _ = _teacher(gen_output)

                        ''' knowledge distillation loss '''
                        t_logit_merge += teacher_logit * torch.tensor(expand_weight, dtype=torch.float32, device=self.device)

                loss_KD = F.kl_div(F.log_softmax(student_logit, dim=1)/self.T, F.softmax(t_logit_merge, dim=1)/self.T, reduction='batchmean')
                loss_KD_total.append(loss_KD.item())

                optimizer_server.zero_grad()
                loss_KD.backward()
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer_server.step()

            val_acc = self.evaluate(global_model, self.valLoader)
            if val_acc > state["best_val_acc"]:
                state["best_val_acc"] = val_acc
                state["best_server_model"] = copy.deepcopy(global_model.state_dict())
                state["best_generator"] = copy.deepcopy(generator.state_dict())

            pbar.set_postfix({
                "loss_KD": np.mean(loss_KD_total),
                "loss_G": np.mean(loss_G_total),
                "loss_con": np.mean(loss_con_total),
                "loss_cls": np.mean(loss_cls_total),
                "loss_div": np.mean(loss_div_total),
                "val_acc": val_acc
            })

        # self.scheduler_cgan.step()
        # scheduler_server.step()

        # restore the best model
        generator.load_state_dict(state["best_generator"])
        global_model.load_state_dict(state["best_server_model"])
        global_model.eval()
        print("[Server | E. Distillation] Best val_acc: {:.4f}".format(state["best_val_acc"]))

        test_acc = self.evaluate(global_model, self.testLoader)
        print("[Server | Generative Knowledge Distilaltion]  After test_acc: {:.4f}".format(test_acc))
        return np.mean(loss_G_total), np.mean(loss_con_total), np.mean(loss_cls_total), np.mean(loss_div_total), np.mean(loss_KD_total), state["best_val_acc"], global_model.state_dict().values()


    def generate_labels(self, number, cls_num):
        labels = np.arange(number)
        proportions = cls_num / cls_num.sum()
        proportions = (np.cumsum(proportions) * number).astype(int)[:-1]
        labels_split = np.split(labels, proportions)
        for i in range(len(labels_split)):
            labels_split[i].fill(i)
        labels = np.concatenate(labels_split)
        np.random.shuffle(labels)
        return labels.astype(int)


    def get_label_weights_from_knowledge_pool(self, knowledge_pool):
        MIN_SAMPLES_PER_LABEL = 1
        label_weights = np.zeros((self.args.num_classes, len(knowledge_pool)))

        for i, (sig, value) in enumerate(knowledge_pool.items()):
            for label, num_data in value["data_summary"].items():
                label_weights[label, i] += num_data  

        qualified_labels = np.where(label_weights.sum(axis=1) >= MIN_SAMPLES_PER_LABEL)[0]
        for i in range(self.args.num_classes):
            if np.sum(label_weights[i], axis=0) > 0:                    # avoid division by zero (lack of data at current round)
                label_weights[i] /= np.sum(label_weights[i], axis=0)
            else:
                label_weights[i] = 0

        # print("label_weights:", label_weights)
        # label_weights = label_weights.reshape((self.num_classes, -1))
        return label_weights, qualified_labels


    def combine_weights(self, weights_1, weights_2):
        ''' Combine two weight tensors '''
        weight = weights_1 + weights_2

        if weight.ndim > 1:
            for i in range(len(weight)):
                if np.sum(weight[i], axis=0) > 0:                    # avoid division by zero (lack of data at current round)
                    weight[i] /= np.sum(weight[i], axis=0)
                else:
                    weight[i] = 0
        else:
            weight /= np.sum(weight, axis=0)

        # print("weights_1:\n", weights_1)
        # print("weights_2:\n", weights_2)
        # print("weight:\n", weight)
        # input()
        return weight


    def get_age_weight(self,):
        ''' Calculate the age weight for each model in the knowledge pool '''
        age_weights = np.zeros(len(self.knowledge_pool))
        for idx, (sig, value) in enumerate(self.knowledge_pool.items()):
            if value["online_age"] > 0:
                age_weights[idx] = math.exp(value["online_age"] * self.online_rate)
            else:
                age_weights[idx] = math.exp(value["offline_age"] * self.offline_rate)

        ''' Normalize the age weight '''
        age_weights /= np.sum(age_weights, axis=0)
        return age_weights
    

    # def get_performance_weight(self,):
    #     ''' Calculate the performance weight for each model in the knowledge pool '''
    #     perf_weights = np.zeros(len(self.knowledge_pool))
    #     class_perf_weights = np.zeros((self.num_classes, len(self.knowledge_pool)))

    #     for idx, (sig, value) in enumerate(self.knowledge_pool.items()):
    #         perf_weights[idx] = value["performance"]
    #         for label, acc in value["class_performance"].items():
    #             class_perf_weights[label][idx] = acc
        
    #     ''' Normalize the performance weight '''
    #     perf_weights /= np.sum(perf_weights, axis=0)

    #     ''' Normalize the class performance '''
    #     for i in range(self.args.num_classes):
    #         if np.sum(class_perf_weights[i], axis=0) > 0:                    # avoid division by zero (lack of data at current round)
    #             class_perf_weights[i] /= np.sum(class_perf_weights[i], axis=0)
    #         else:
    #             class_perf_weights[i] = 0

    #     # print("perf_weights:", perf_weights)
    #     # input()
    #     return perf_weights, class_perf_weights


    def get_importance_weight(self,):
        ''' Importance weight for each model can be approximated by the sum of the label weights '''
        label_weights, _ = self.get_label_weights_from_knowledge_pool(self.knowledge_pool)
        importance_weights = np.sum(label_weights, axis=0)

        ''' Normalize the importance weight '''
        importance_weights /= np.sum(importance_weights, axis=0)
        # print("importance_weights:", importance_weights)
        # input()
        return importance_weights


    def update_knowledge_pool(self, client_list, rounds):
        for client in client_list:
            status = client.status
            acc = client.test()["test_acc"]
            # print("client {} | status: {} | performance: {:.4f}".format(client.cid, status, acc))

            signature = "{cid}-{data}".format(cid=client.cid, data=str(list(client.data_distribution.keys())))
            # print(client.data_distribution)

            if signature in self.knowledge_pool:                                    # update the snapshot in knowledge pool
                item = self.knowledge_pool[signature]
                if status == "online":
                    item["model"] = copy.deepcopy(client.model)
                    item["data_summary"] = client.data_distribution
                    item["num_data"] = len(client.trainLoader.dataset)
                    item["performance"] = acc
                    item["online_age"] += 1
                    item["offline_age"] = 0
                else:
                    item["online_age"] = 0
                    item["offline_age"] += 1
            else:
                if status == "online":                                              # new to the knowledge pool
                    self.knowledge_pool[signature] = {
                        "model": copy.deepcopy(client.model),
                        "data_summary": client.data_distribution,
                        "num_data": len(client.trainLoader.dataset),
                        "performance": acc,
                        "online_age": 1,
                        "offline_age": 0,
                    }


    def show_knowledge_pool(self,):
        print("\n-----> Knowledge Pool Status, Size: {}".format(len(self.knowledge_pool)))
        for key, value in self.knowledge_pool.items():
            line_head = Fore.YELLOW + "[Tag: {}]".format(key)
            line_mid1 = "Acc.: {:.4f}".format(value["performance"]) + Fore.RESET
            line_mid2 = "{}On. Age: {}".format(Fore.GREEN if value["online_age"] > 0 else Fore.LIGHTBLACK_EX, value["online_age"]) + Fore.RESET
            line_tail = "{}Off. Age: {}".format(Fore.RED if value["offline_age"] > 0 else Fore.LIGHTBLACK_EX, value["offline_age"]) + Fore.RESET
            print("{:<55} {:<25} {:<26} {:<27}".format(line_head, line_mid1, line_mid2, line_tail))
        print("-" * 101 + "\n")


    def aggregate_from_knowledge_pool(self, rounds) -> np.ndarray:
        ''' Aggregate with all models in the knowledge pool (no selection) '''
        client_numData_model_pair = [
            (value["model"], value["num_data"]) for sig, value in self.knowledge_pool.items()
        ]

        # Create a list of weights, each multiplied by the related number of examples
        model_weights = [
            [model.state_dict()[params] for params in model.state_dict()] for model, _ in client_numData_model_pair
        ]

        # Calculate the totol number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in client_numData_model_pair])
        # print("num_examples_total:", num_examples_total)

        # Calculate the weights of each model
        dataset_weights = [
            num_examples / num_examples_total for _, num_examples in client_numData_model_pair
        ]

        # Get the age weight of each model
        age_weights = self.get_age_weight()

        # Get the importance weight of each model
        importance_weights = self.get_importance_weight()

        # Calculate the final weights
        final_weights = self.age_ld * (torch.tensor(age_weights, device=self.device) * torch.tensor(dataset_weights, device=self.device)) + \
                         + self.impt_ld * torch.tensor(importance_weights, device=self.device)
        final_weights /= torch.sum(final_weights)

        # print("Age_weights:", age_weights)
        # print("Perf_weights:", perf_weights)
        # print("Datset_weights:", dataset_weights)
        # print("Final_weights:", final_weights)
        # input()

        # Compute average weight of each layer using the findal weights
        weights_prime = [
            reduce(torch.add, [w * weight for w, weight in zip(layer_updates, final_weights)])
            for layer_updates in zip(*model_weights)
        ]

        # Compute average weight of each layer
        # weights_prime = [
        #     reduce(torch.add, layer_updates)
        #     for layer_updates in zip(*model_weights)
        # ]
        return weights_prime
    

    def evaluate(self, server_model, testLoader) -> Tuple[float, float]:
        avg_acc = []
        for name, loader in testLoader.items():
            # _, acc_indi, _ = self.fl_strategy._test(server_model, loader)
            acc_indi = self.fl_strategy._test(server_model, loader)["test_acc"]
            avg_acc.append(acc_indi)
        return np.mean(avg_acc)


    # def _test(self, model, testLoader) -> Tuple[float, float]:
    #     ''' Test function for the client '''
    #     total_loss, total_correct = [], []
    #     num_classes = self.args.num_classes
    #     correct_predictions = {i: 0 for i in range(num_classes)}
    #     total_counts = {i: 0 for i in range(num_classes)}

    #     model.eval()
    #     with torch.no_grad():
    #         for x, label in testLoader:
    #             x, label = x.to(self.device), label.to(self.device)
    #             output, _ = model(x)
    #             predict = torch.argmax(output.data, 1)
    #             total_correct.append((predict == label).sum().item() / len(predict))
    #             total_loss.append(F.cross_entropy(output, label).item())

    #             for i in range(num_classes):
    #                 mask = label == i
    #                 correct_predictions[i] += (predict[mask] == label[mask]).sum().item()
    #                 total_counts[i] += mask.sum().item()

    #     class_acc = {i: (correct_predictions[i] / total_counts[i]) if total_counts[i] > 0 else 0 for i in range(num_classes)}
    #     return np.mean(total_loss), np.mean(total_correct), class_acc