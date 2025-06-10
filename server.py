from typing import Dict, List, Tuple
from collections import OrderedDict
from colorama import Fore, Style
import numpy as np
import wandb
import copy
import datetime
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from utils.base_client import BaseClient
from client import FLclient
from strategies.strategy import Strategy
from strategies.feddpfl import FedDPFL
from strategies.mifa import MIFA
from utils.data_model import DataModel
from utils.util import *



''' FL Server class '''
class FLserver(BaseClient):
    def __init__(self, clients: List[FLclient], testset: Dict[str, Dataset], strategy: Strategy, device: torch.device, params: Dict, args: Dict, sim=None):
        super().__init__(device, args)
        '''
        clients: a list of current clients
        dataLoader: dataloader for testing data
        dataLoader_indi: list of dataloader for individual classes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        device: cpu or gpu
        args: arguments
        '''
        self.args = args
        self.client_list = clients
        self.active_clients = []
        self.inactive_clients = []
        self._current_tid = 0
        self.testset = self.init_testset(testset)
        self.testLoader = None

        self.strategy = copy.deepcopy(strategy)
        self._strategy = strategy.__class__.__name__
        self.global_info = {}

        self.writer = SummaryWriter(args.log_dir + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), flush_secs=10)
        self.params = params
        self.device = device
        self.sim = sim
        

    def init_testset(self, testset):
        if self.args.incremental_type == "class-incremental":
            tmp_dataset = {}
            for name, dataset in testset.items():
                tmp_dataset[name] = DataModel().divide_dataset_to_incremental_partition(dataset, self.args)

            task_dataset = {
                f"task_{i}": {
                    name: tmp_dataset[name][f"task_{i}"] for name in testset.keys()
                }
                for i in range(1, 6)
            }
            return task_dataset
        else:
            return testset


    def get_current_dataLoader(self,):
        if self.args.incremental_type == "class-incremental":
            testset = {
                name: ConcatDataset([self.testset[f"task_{i}"][name] for i in range(1, self._current_tid + 1)])
                for name in self.testset["task_1"].keys()
            }
        else:
            testset = self.testset
    
        testLoader = {
            name: DataLoader(testset, batch_size=self.args.batch_size, pin_memory=True)
            for name, testset in testset.items()
        }
        return testLoader


    def update_incremental_data(self, rounds):
        ''' Update the server's data and client's data if is under class-incremental learning '''
        self._current_tid += 1
        self.testLoader = self.get_current_dataLoader()
        self.strategy.prev_global_model = copy.deepcopy(self.model)

        for i in range(len(self.client_list)):
            self.client_list[i].update_incremental_data()

        if rounds > 1:
            if self._strategy == "Target":
                self.strategy.data_generation(global_model=self.model, rounds=self._current_tid - 1)
                self.strategy.syn_data_loader = self.strategy.get_syn_data_loader(rounds=self._current_tid - 1)
                self.strategy.syn_iter = iter(self.strategy.syn_data_loader)


    def add_client(self, client: FLclient):
        ''' Add a client to the server '''
        self.client_list.append(client)
        self.set_client_model(client.cid, self.get_parameters())
        self.active_clients.append(client.cid)
        print("[+] Add client, cid:{}".format(client.cid))
        # print("Total clients: {}, {}".format(len(self.client_list), [c.cid for c in self.client_list]))


    def set_client_model(self, cid, parameters):
        ''' Set the model of the client '''
        for i in range(len(self.client_list)):
            if self.client_list[i].cid == cid:
                self.client_list[i].set_parameters(parameters)
                break


    ''' -------------------------------------------------- Core Functions -------------------------------------------------- '''

    def initialization(self,):
        self.active_clients = np.sort(self.active_clients)

        ''' Initialize the server's and client's local dataset '''
        self.update_incremental_data(rounds=0)

        ''' Initialization according to the strategy '''
        self.strategy._initialization(client_list=self.client_list, active_clients=self.active_clients, global_model=self.model, global_info=self.global_info)
        
        ''' If DPFL, initialize the knowledge pool '''
        if self.args.dpfl:
            # self.mifa = MIFA(self.model, self.device, self.args)
            self.knowledge_pool = FedDPFL(self.strategy, self.device, self.args, self.params)

            ''' Create lr_scheduler for global model '''
            self.optimizer_server = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.scheduler_server = torch.optim.lr_scheduler.StepLR(self.optimizer_server, step_size=1, gamma=0.998)

            self.knowledge_pool.testLoader = copy.deepcopy(self.testLoader)

            ''' Get client's validation data '''
            valset = [self.client_list[i].valLoader.dataset for i in range(len(self.client_list))]
            self.knowledge_pool.valLoader = {
                "total_val": DataLoader(ConcatDataset(valset), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
            }
            print(Fore.YELLOW + "\n[Server] Initializing the knowledge pool..." + Fore.RESET)
            print(Fore.YELLOW + "Optimized Parameters for DPFL: {}".format(self.params) + Fore.RESET)

        print("\n" + Fore.RED + "[Server] `{}` initialization done".format(self._strategy) + Fore.RESET)


    def train_clients(self, rounds: int):
        ''' Train all clients with status "online" '''
        print(Fore.CYAN + "[Server] Start training client" + Fore.RESET)
        for i in self.active_clients:
            if self.sim:
                self.sim.process(lambda sim=self.sim, i=i, r=rounds: self.strategy._server_train_func(cid=i, rounds=rounds, client_list=self.client_list, global_model=self.model, global_info=self.global_info))
            else:
                self.strategy._server_train_func(cid=i, rounds=rounds, client_list=self.client_list, global_model=self.model, global_info=self.global_info)

        ''' Ensure all scheduled training finishes in this virtual round '''
        if self.sim:
            self.sim.run()  

        ''' DPFL: update the local models to knowledge pool '''
        # self.mifa.update_mifa(client=self.client_list[i], global_model=self.model)
        # if self.args.dpfl:
        #     self.knowledge_pool.update_knowledge_pool(client=self.client_list[i], rounds=rounds)


    def aggregate_clients(self, rounds: int):
        ''' Aggregation function of different strategies '''
        new_weights = self.strategy._server_agg_func(rounds, self.client_list, self.active_clients, self.model)

        ''' DFPL: Aggregate from the knowledge pool '''
        if self.args.dpfl:
            ''' MIFA '''
            # new_weights = self.mifa.aggregation_mifa(global_model=self.model, rounds=rounds)

            self.knowledge_pool.update_knowledge_pool(client_list=self.client_list, rounds=rounds)
            self.knowledge_pool.show_knowledge_pool()
            
            ''' aggregate from selecting models in the knowledge pool '''
            agg_weights = self.knowledge_pool.aggregate_from_knowledge_pool(rounds=rounds)
            
            ''' set global model's weight for ensemble distillation '''
            self.set_parameters(agg_weights)
            loss_g, loss_con, loss_cls, loss_div, loss_kd, avg_acc, new_weights_prime = self.knowledge_pool.data_free_knowledge_distillation(
                global_model=self.model, 
                optimizer_server=self.optimizer_server,
                scheduler_server=self.scheduler_server,
                rounds=rounds,
            )

            # result = {"loss_g": loss_g, "loss_con": loss_con, "loss_cls": loss_cls, "loss_div": loss_div, "loss_kd": loss_kd, "avg_acc": avg_acc}
            # wandb.log(result)
            print("[Server] loss_KD: {:.4f}, loss_G: {:.4f}, loss_con: {:.4f}, loss_cls: {:.4f}, loss_div: {:.4f}".format(loss_kd, loss_g, loss_con, loss_cls, loss_div))

            # merge the new_weights and new_weights_prime
            # a = 0.5
            # new_weights_prime = [a * w + (1 - a) * wp for w, wp in zip(new_weights, new_weights_prime)]

            ''' rounds <= args.round_start '''
            if (self.args.dynamic_type in ["incremental-arrival", "incremental-departure"] and rounds <= 50) \
                or (self.args.dynamic_type in ["round-robin"] and rounds <= 10):
                print(Fore.CYAN + "[Server] Not updating the global model by DPFL" + Fore.RESET)
                pass
            else:
                new_weights = new_weights_prime

        self.set_parameters(new_weights)                                        # set global model's weight
        for i in self.active_clients:                                           # set client model's weight if status == "online"
            self.set_client_model(cid=i, parameters=new_weights)
        print(Fore.CYAN + "[Server] Done aggregating client models" + Fore.RESET)

    ''' -------------------------------------------------- Core Functions -------------------------------------------------- '''


    def set_client_state(self, active_ids: List[int], inactive_ids: List[int]):
        self.set_active_client(active_ids)
        self.set_inactive_client(inactive_ids)


    def set_inactive_client(self, clients: List[int]):
        for i in range(len(self.client_list)):
            if self.client_list[i].cid in clients:
                self.client_list[i].status = "offline"

        self.inactive_clients = np.sort(clients)
        self.active_clients = np.sort([c for c in self.active_clients if c not in clients])
        print("[-] set inactive: {}".format(self.inactive_clients))


    def set_active_client(self, clients: List[int]):
        for i in range(len(self.client_list)):
            if self.client_list[i].cid in clients:
                self.client_list[i].status = "online"
                # if self._strategy != "FedProto":
                self.set_client_model(self.client_list[i].cid, self.get_parameters())   # 恢復訓練的 client 拿到最新的 model

        self.active_clients = np.sort(clients)
        self.inactive_clients = np.sort([c for c in self.inactive_clients if c not in clients])
        print("[+] set active: {}".format(self.active_clients))


    def show_clients(self,):
        # print("Active: {}".format(self.active_clients))
        # print("Inactive clients: {}".format(self.inactive_clients))
        bar = "Client |"
        for i in range(len(self.client_list)):
            bar += " {} |".format(i)
        bar += "\n"
        line = "-" * len(bar) + "\n"
        content = "Status |"
        for i in range(len(self.client_list)):
            if self.client_list[i].status == "online":
                content += Fore.LIGHTRED_EX + " O " + Fore.RESET + "|"
            else:
                content += Fore.LIGHTBLACK_EX + " X " + Fore.RESET + "|"
        content += "\n"
        print(line + bar + line + content + line)


    def evaluate(self, rounds: int, model=None, tag="all"):
        # if self._strategy == "FedProtoO":
            # ''' Evaluate each client model on every domains '''
            # print("===== Evaluate each client")
            # result = self.evaluate_each_client()
        # else:

        ''' Evaluate the global model on individual domains '''
        print("===== Evaluate each domain")

        if tag == "all":
            ''' Test performance on all models '''
            all_model = copy.deepcopy(self.model)
            all_model_params = self.strategy._aggregation(self.client_list, mode="all")
            set_param(model=all_model, parameters=all_model_params)
            result = self.evaluate_each_domain(all_model)
        elif tag == "active":
            ''' Test performance on active models '''
            result = self.evaluate_each_domain(model)
        else:
            raise ValueError("Non-supported tag")
            

        total_loss, total_acc = [], []
        total_class_acc = {i: [] for i in range(self.args.num_classes)}
        for name in self.testLoader.keys():
            loss_indi, acc_indi, class_indi = result[name]["loss"], result[name]["acc"], result[name]["class_acc"]
            total_loss.append(loss_indi)
            total_acc.append(acc_indi)
            for class_id in class_indi:
                total_class_acc[class_id].append(class_indi[class_id])
                wandb.log({
                    f"({name}) {class_id}_{tag}": class_indi[class_id],
                    "rounds": rounds,
                })
                # self.writer.add_scalar(f"({name}) {class_id}_{tag}", class_indi[class_id], rounds)
                
            print(Fore.GREEN + "[Server, {}] test_loss_{}: {:.4f}, test_acc_{}: {:.4f}".format(name, tag, loss_indi, tag, acc_indi) + Fore.RESET)
            wandb.log({
                f"({name})_loss_{tag}": loss_indi,
                f"({name})_acc_{tag}": acc_indi,
                "rounds": rounds,
            })
            # self.writer.add_scalar(f"({name})_loss_{tag}", loss_indi, rounds)
            # self.writer.add_scalar(f"({name})_acc_{tag}", acc_indi, rounds)

        
        ''' Calculate the average loss & accuracy '''
        total_loss = np.mean(total_loss)
        total_acc = np.mean(total_acc)
        print(Fore.GREEN + Style.BRIGHT + "[Server] Total_loss_{}: {:.4f}, Total_acc_{}: {:.4f}".format(tag, total_loss, tag, total_acc) + Style.RESET_ALL + Fore.RESET)
        wandb.log({
            f"total_loss_{tag}": total_loss,
            f"total_acc_{tag}": total_acc,
            "rounds": rounds,
        })
        self.writer.add_scalar(f"{tag}/total_loss", total_loss, rounds)
        self.writer.add_scalar(f"{tag}/total_acc", total_acc, rounds)
        self.writer.flush()

        ''' Calculate each class's accuracy '''
        total_class_acc = {i: np.mean(total_class_acc[i]) for i in range(self.args.num_classes)}
        for class_id in total_class_acc:
            wandb.log({
            f"total_acc {class_id}_{tag}": total_class_acc[class_id],
            "rounds": rounds,
        })
        return result


    def evaluate_each_domain(self, model):
        ''' Evaluate the global model on individual domains
        Returns:
            result: {
                "MNIST": {
                    "loss": 0.1,
                    "acc": 0.9,
                    "class": {
                        0: 0.99,
                        1: 0.98,
                    }
                },
                "USPS": {
                    "loss": 0.2,
                    "acc": 0.8,
                },
            }
        '''
        result = {}
        for name, loader in self.testLoader.items():
            result_indi = self.test(model, loader)
            loss_indi, acc_indi, class_indi = result_indi["test_loss"], result_indi["test_acc"], result_indi["class_acc"]
            record = {
                "loss": loss_indi,
                "acc": acc_indi,
                "class_acc": class_indi,
            }
            result[name] = record
        return result
    

    def evaluate_each_client(self,):
        ''' Evaluate each client on testing data '''
        result = {
            name: {
                "loss": [],
                "acc": [],
                "class_acc": {
                    i: [] for i in range(self.args.num_classes)
                },
            } for name in self.testLoader.keys()
        }
    
        for i in tqdm(range(len(self.client_list)), leave=False):
            # if self.client_list[i].status == "online":
                client_result = self.evaluate_each_domain(self.client_list[i].model)
                for name in client_result:
                    result[name]["loss"].append(client_result[name]["loss"])
                    result[name]["acc"].append(client_result[name]["acc"])
                    for class_id in client_result[name]["class_acc"]:
                        result[name]["class_acc"][class_id].append(client_result[name]["class_acc"][class_id])

        for name in result:
            result[name]["loss"] = np.mean(result[name]["loss"])
            result[name]["acc"] = np.mean(result[name]["acc"])
            for class_id in result[name]["class_acc"]:
                result[name]["class_acc"][class_id] = np.mean(result[name]["class_acc"][class_id])
        return result


    def test(self, model,  dataLoader: DataLoader):
        if self._strategy == "FedProto" or self._strategy == "FedProtoV2":
            return self.strategy._test(model, dataLoader, global_protos=self.strategy.global_protos)
        else:
            return self.strategy._test(model, dataLoader)