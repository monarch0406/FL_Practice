import argparse
import datetime
import wandb
import torch
import random
import numpy as np
from colorama import Fore, Style
from server import FLserver
from client import FLclient
from strategies.fedavg import FedAvg
from model import MyNet
from utils.util import *
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser("FL Dynamic-Client-Participation")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.003, type=float)
parser.add_argument("--val_ratio", default=0.2, type=float)
parser.add_argument("--sample_ratio", default=0.1, type=float, help="sample ratio of local data")
parser.add_argument("--num_clients", default=1, type=int, help="number of clients")
parser.add_argument("--num_rounds", default=10, type=int, help="number of rounds")
parser.add_argument("--dynamic_round", default=1, type=int)
parser.add_argument("--dynamic_type", default="random", type=str)
parser.add_argument("--num_epochs", default=5, type=int,help="number of local epochs")
parser.add_argument("--algorithm", default="fedavg", type=str)
parser.add_argument("--static", action="store_true")
parser.add_argument("--figure_path", default="./figures/", type=str)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

os.makedirs(args.figure_path, exist_ok=True)

seed_everything(seed=args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: `{device}`")


''' Load data '''
datasets = ["MNIST", "USPS", "SVHN", "SYN"]
trainsets, valsets, testLoaders = {}, {}, {}
for name in datasets:
    trainsets[name], valsets[name], testLoaders[name] = load_data(dataset_name=name, args=args)


base_model = MyNet()
first_domain = ["MNIST", "USPS", "SVHN", "SYN"]
second_domain = ["MNIST", "USPS", "SVHN", "SYN"]

imm_matrix = np.zeros((len(first_domain), len(second_domain)))
matrix = np.zeros((len(first_domain), len(second_domain)))

for m, domain_1 in enumerate(first_domain):
    for n, domain_2 in enumerate(second_domain):
        ''' Wandb for training logging '''
        run = wandb.init(
            project="FL-Dynamic-Client-Participation",
            name="Fixed ({d1}, {d2}) num_clients={nc}-num_epochs={ne}, {time}"
                .format(d1=domain_1, d2=domain_2, nc=args.num_clients, ne=args.num_epochs, time=datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")),
            config=args,
        )

        ''' Create FL Strategy instance '''
        strategy = FedAvg()

        ''' Create FL Server instance '''
        server = FLserver(clients=[], dataLoader=testLoaders, strategy=strategy, device=device, args=args)
        # server 每次起始的 model 都是一樣的
        server.model.load_state_dict(torch.load("./checkpoint/{}_10.pth".format(domain_1)))

        ''' Create FL Clients instance'''
        for i in range(args.num_clients):
            domain = domain_2
            client = FLclient(
                        cid=i, 
                        trainset=trainsets[domain], 
                        valset=valsets[domain],
                        domain=domain,
                        strategy=strategy,
                        device=device,
                        args=args,
                    )
            server.add_client(client)

        for i in range(1, args.num_rounds+1):
            print("\n----------- Round {}/{} -----------".format(i, args.num_rounds))
            server.show_clients()
            server.train_clients()
            server.aggregate_clients()
            result = server.evaluate_individual_domain(rounds=i)

            if i == 1:
                imm_matrix[m][n] = result[domain_1]
        matrix[m][n] = result[domain_1]

        print("imm_matrix:\n", imm_matrix)
        print("matrix:\n", matrix)

        wandb.finish()

label = ["MNIST", "USPS", "SVHN", "SYN"]
# imm_matrix = np.array([
#     [0.9867, 0.5281, 0.4315, 0.1787],
#     [0.7808, 0.8788, 0.6933, 0.3415],
#     [0.6228, 0.5542, 0.8424, 0.4491],
#     [0.8299, 0.4863, 0.6911, 0.8611],
# ])

plt.figure(figsize=(8, 6))
ax = sns.heatmap(imm_matrix.T, cmap="Blues", annot=True)
ax.set_title("Accuracy after domain change")
ax.set_xticklabels(label)
ax.set_yticklabels(label)
plt.savefig("./figures/imm.png")

# final_matrix = np.array([
#     [0.9913, 0.5701, 0.3885, 0.1707],
#     [0.7708, 0.9463, 0.6289, 0.3012],
#     [0.5591, 0.6001, 0.8622, 0.32],
#     [0.728, 0.5679, 0.686, 0.8775],
# ])

plt.figure(figsize=(8, 6))
ax = sns.heatmap(matrix.T, cmap="Blues", annot=True)
ax.set_title("Accuracy at final")
ax.set_xticklabels(label)
ax.set_yticklabels(label)
plt.savefig("./figures/final.png")