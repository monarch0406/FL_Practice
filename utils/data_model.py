import argparse
from argparse import ArgumentParser
from collections import Counter
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset, RandomSampler
from torchvision.datasets import MNIST, USPS, SVHN, EMNIST, CIFAR10, CIFAR100, FER2013, Caltech101, ImageFolder, Flickr8k
from utils.mnist_m import MNISTM
from utils.myDatasets import Dataset2Memory
from utils.flickrDataset import Flickr8kDataset
from torchvision.utils import make_grid, save_image
from utils.util import plot_distribution, bubble_plot_distribution


# parser = argparse.ArgumentParser("DCP Non-IID Data Preparation")
# parser.add_argument("--batch_size", default=128, type=int)
# parser.add_argument("--val_ratio", default=0.1, type=float)
# parser.add_argument("--num_clients", default=5, type=int, help="number of clients")
# parser.add_argument("--num_classes", default=10, type=int, help="number of classes")
# parser.add_argument("--num_domains", default=4, type=int, help="number of domains")
# parser.add_argument("--skew_type", default="label", type=str, help="label, feature, quantity")
# parser.add_argument("--label_alpha", default=100, type=float, help="concentration level of Dirichlet distribution")
# parser.add_argument("--feature_alpha", default=0.5, type=float, help="concentration level of Dirichlet distribution")
# parser.add_argument("--quantity_alpha", default=1.0, type=float, help="concentration level of Dirichlet distribution")
# parser.add_argument("--seed", default=42, type=int)
# args = parser.parse_args()


''' Convert a grayscale image to rgb '''
class GrayscaleToRgb:
    def __call__(self, image):
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.dstack([image, image, image])
        return Image.fromarray(image)


class DataModel:
    def __init__(self):
        pass

    def load_data(self, dataset_name, args):
        if dataset_name == "MNIST":
            transform = transforms.Compose([
                GrayscaleToRgb(),
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            dataset = MNIST("./data/digit/", download=True, train=True, transform=transform)
            testset = MNIST("./data/digit/",  download=True, train=False, transform=transform)
        elif dataset_name == "USPS":
            transform = transforms.Compose([
                GrayscaleToRgb(),
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            dataset = USPS("./data/digit/", download=False, train=True, transform=transform)
            testset = USPS("./data/digit/", download=False, train=False, transform=transform)
        elif dataset_name == "SVHN":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            dataset = SVHN("./data/digit/", download=False, split="train", transform=transform)
            testset = SVHN("./data/digit/", download=False, split="test", transform=transform)
        elif dataset_name == "SYN":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            dataset = ImageFolder(root="./data/digit/syn/train/", transform=transform)
            testset = ImageFolder(root="./data/digit/syn/val/", transform=transform)
        elif dataset_name == "CIFAR10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            dataset = CIFAR10("./data", download=True, train=True, transform=transform)
            testset = CIFAR10("./data", download=True, train=False, transform=transform)
        elif dataset_name == "CIFAR100":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            dataset = CIFAR100("./data", download=False, train=True, transform=transform)
            testset = CIFAR100("./data", download=False, train=False, transform=transform)
        elif dataset_name == "fer2013":
            transform = transforms.Compose([
                GrayscaleToRgb(),
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # dataset = FER2013("./data/", split="train", transform=transform)
            # testset = FER2013("./data/", split="test", transform=transform)

            Dataset = ImageFolder(root="./data/fer2013/train/", transform=transform)
            testset = ImageFolder(root="./data/fer2013/test/", transform=transform)

            # sample to equalize the number of samples in each class
            class_counts = Counter(Dataset.targets)
            min_count = min(class_counts.values())
            sampled_indices = []
            for label in class_counts:
                indices = [i for i in range(len(Dataset)) if Dataset.targets[i] == label]
                sampled_indices.extend(random.sample(indices, min_count))
            dataset = Subset(Dataset, sampled_indices)
            # testset = Subset(testset, sampled_indices)
            # print("Training distribution:", Counter([label for _, label in dataset]))


            # select specific classes
            # {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
            dataset = Subset(dataset, [i for i in range(len(dataset)) if Dataset.targets[dataset.indices[i]] in [0, 3, 4]])
            testset = Subset(testset, [i for i in range(len(testset)) if testset.targets[i] in [0, 3, 4]])

            # print("\nTraining distribution:", Counter([label for _, label in dataset]))
            # print("\n esting distribution:", Counter([label for _, label in testset]))

            # print("Dataset size:", len(dataset))
            # print("Testset size:", len(testset))
            # print("Dataset classes:", dataset.dataset.classes)
            # input("Press Enter to continue...")
        elif dataset_name == "EMNIST":
            transform = transforms.Compose([
                GrayscaleToRgb(),
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            Dataset = EMNIST("./data", download=True, split="mnist", train=True, transform=transform)
            testset = EMNIST("./data", download=True, split="mnist", train=False, transform=transform)

            train_len = int(len(Dataset) * 0.1)
            dataset, _ = random_split(Dataset, [train_len, len(Dataset) - train_len], torch.Generator().manual_seed(0))

        elif dataset_name == "MNISTM":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            dataset = MNISTM("./data/digit/", download=False, train=True, transform=transform)
            testset = MNISTM("./data/digit/", download=False, train=False, transform=transform)

        elif dataset_name == "Caltech101":
            transform = transforms.Compose([
                GrayscaleToRgb(),
                transforms.ToTensor(),
                transforms.Resize((32, 32), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            Dataset = Caltech101("./data", download=False, transform=transform)
            # split the dataset into training and testing sets
            train_len = int(len(Dataset) * 0.9)
            dataset, testset = random_split(Dataset, [train_len, len(Dataset) - train_len], torch.Generator().manual_seed(args.seed))

        elif dataset_name in ["Caltech", "Amazon", "DSLR", "Webcam"]:
            transform = transforms.Compose([
                transforms.Resize((32, 32), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            Dataset = ImageFolder(root=f"./data/office_caltech_10/{dataset_name}/", transform=transform)
            # split the dataset into training and testing sets
            train_len = int(len(Dataset) * 0.8)
            dataset, testset = random_split(Dataset, [train_len, len(Dataset) - train_len], torch.Generator().manual_seed(args.seed))

        elif dataset_name == "custom":
            if not hasattr(args, "custom_data_path") or not args.custom_data_path:
                raise ValueError("Please provide a valid custom_data_dir in args")
            
            transform = transforms.Compose([
                # transforms.CenterCrop(32),
                transforms.Resize((32, 32), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # split the dataset into training and testing sets
            Dataset = ImageFolder(root=args.custom_data_path, transform=transform)
            labels = [label for _, label in Dataset]

            # perform stratified split
            train_idx, test_idx = train_test_split(
                range(len(Dataset)),
                test_size=0.2,
                stratify=labels,
                random_state=args.seed
            )
            dataset = Dataset
            # dataset = Subset(Dataset, train_idx)
            testset = Subset(Dataset, test_idx)

            # print("Training distribution:", Counter([label for _, label in dataset]))
            # print("Testing distribution:", Counter([label for _, label in testset]))

        else:
            raise ValueError("Non-supported dataset")
        
        if args.load_data_to_memory:
            dataset = Dataset2Memory(dataset_name=dataset_name, dataset=dataset, train=True, args=args)
            testset = Dataset2Memory(dataset_name=dataset_name, dataset=testset, train=False, args=args)
        return dataset, testset


    def load_noniid_partition(self, dataset, args):
        random.seed(args.seed+2)
        np.random.seed(args.seed+2)
        num_clients = args.num_clients
        num_classes = args.num_classes
        num_domains = args.num_domains
        skew_type = args.skew_type
        alpha = args.alpha
        # feature_alpha = args.feature_alpha
        # quantity_alpha = args.quantity_alpha

        ''' Generate label/feature distribution & quantity skew using Dirichlet distribution '''
        label_distribution, feature_distribution, quantity_distribution = None, None, None
        if skew_type == "label":
            label_distribution = np.random.dirichlet(np.repeat(alpha, num_clients), size=num_classes).T
            # plot_distribution(label_distribution, alpha, mode="label", args=args)
            bubble_plot_distribution(label_distribution, alpha, mode="label", args=args)
        elif skew_type == "feature":
            feature_distribution = np.random.dirichlet(np.repeat(alpha, num_clients), size=num_domains).T
            # plot_distribution(feature_distribution, alpha, mode="feature", args=args)
        elif skew_type == "quantity":
            quantity_distribution = np.random.dirichlet(np.repeat(alpha, num_clients), size=1).T
            # plot_distribution(quantity_distribution, alpha, mode="quantity", args=args)
        else:
            raise ValueError("Invalid skew type: {}".format(skew_type))
        
        # print(label_distribution, label_distribution.shape)
        print("[*] Using `{}` skewness".format(skew_type))

        ''' Load data
        dataset_info = {
            "MNIST": {
                0: [...],
                1: [...],
            },
            "USPS": {
                0: [...],
                1: [...],
            }
        }
        '''

        ''' Load data, each with 4 different domains '''
        if dataset == "Mnist":
            DataSet = ["MNIST"]                                 # MNIST
        elif dataset == "Cifar10":
            DataSet = ["CIFAR10"]                               # CIFAR10
        elif dataset == "Digits":
            DataSet = ["MNIST", "USPS", "SVHN", "SYN"]          # Digits
        elif dataset == "Office-Caltech":
            DataSet = ["Amazon", "DSLR", "Webcam", "Caltech"]   # Office-Caltech
        else:
            DataSet = [dataset]
            # raise ValueError("Invalid dataset: {}".format(dataset))
        
        # DataSet = ["CIFAR100"]
        
        datasets, testsets = {}, {}
        dataset_info = {}
        testset_info = {}
        for name in DataSet:
            datasets[name], testsets[name] = self.load_data(dataset_name=name, args=args)

            info = {i: [] for i in range(num_classes)}
            for idx, (img, label) in enumerate(datasets[name]):
                info[label].append(idx)
            dataset_info[name] = info

            test_info = {i: [] for i in range(num_classes)}
            for idx, (img, label) in enumerate(testsets[name]):
                test_info[label].append(idx)
            testset_info[name] = test_info

        # dataset_info = {
        #     "MNIST": {
        #         i: [j for j in range(10)]
        #         for i in range(10)
        #     }
        # }

        dataset_statistics = pd.DataFrame({
            name: {
                label: len(dataset_info[name][label]) for label in dataset_info[name]
            }
            for name in dataset_info
        })
        print("\n----------------------- Dataset statistics: -----------------------\n", dataset_statistics.T)

        ''' Preprocess USPS and SVHN for balancing testing data '''
        # for name in ["USPS", "SVHN"]:
        #     max_samples = int(np.mean([len(testset_info[name][label]) for label in testset_info[name]]))
        #     for label in testset_info[name]:
        #         indices = testset_info[name][label]
        #         if len(indices) > max_samples:
        #             indices = np.random.choice(indices, max_samples, replace=False)
        #             testset_info[name][label] = indices

        balanced_testset = {}
        for name in testset_info:
            selected_indices = [
                idx for label in testset_info[name] for idx in testset_info[name][label]
            ]
            balanced_testset[name] = Subset(testsets[name], selected_indices)

        testset_statistics = pd.DataFrame({
            name: {
                label: len(testset_info[name][label]) for label in testset_info[name]
            }
            for name in testset_info
        })
        print("\n----------------------- Testset statistics: -----------------------\n", testset_statistics.T)



        client_datasets = {i: {} for i in range(num_clients)}
        df_partition = {name: [] for name in DataSet}
        '''
        client_datasets = {
            0: {
                "MNIST": [...],
                "USPS": [...],
            },
            1: {
                "MNIST": [...],
                "USPS": [...],
            },
            ...
        }
        '''

        for idx, dataset_name in enumerate(dataset_info):                               # loop through every dataset (domain)
            ds_info = dataset_info[dataset_name]
            ds_stats = dataset_statistics[dataset_name]
            
            for c in range(num_clients):
                # ---------- Label Imbalance ----------
                if skew_type == "label":
                    client_distribution = label_distribution[c]                         # 取得他的 label distribution
                # --------- Feature Imbalance ---------
                elif skew_type == "feature":
                    client_distribution = feature_distribution[c][idx]                  # 取得他的 feature distribution
                    client_distribution = np.repeat(client_distribution, num_classes)
                # --------- Quantity Imbalance --------
                elif skew_type == "quantity":
                    client_distribution = quantity_distribution[c]                      # 取得他的 quantity distribution
                    client_distribution = np.repeat(client_distribution, num_classes)
                else:
                    raise ValueError("Invalid skew type: {}".format(skew_type))

                # print("\n---------- Client {} ----------".format(c))
                # print("Distribution:", client_distribution)

                # Sample indices based on the specified label distribution
                selected_indices = []
                partition = []
                for label, prob in enumerate(client_distribution):
                    try:
                        num_samples = int(ds_stats[label] * prob)
                        indices = np.random.choice(ds_info[label], num_samples, replace=False)

                        # remove selected indices to avoid overlapping samples
                        ds_info[label] = [l for l in ds_info[label] if l not in indices]
                        
                        # ensure each label has at least one sample to avoid empty dataset
                        if num_samples == 0 and args.incremental_type == "class-incremental":
                            indices = np.random.choice(ds_info[label], 1, replace=False)
                    except ValueError:
                        pass

                    selected_indices.extend(indices)
                    partition.append(len(indices))
                df_partition[dataset_name].append(partition)

                # if client's dataset is empty, randomly select samples from the remaining dataset
                if len(selected_indices) == 0:
                    non_empty_labels = [label for label in ds_info if len(ds_info[label]) > 0]      # 找出還有 sample 的 label
                    random_label = np.random.choice(non_empty_labels, 1, replace=False)[0]          # 隨機選一個 label
                    indices = np.random.choice(ds_info[random_label], 1, replace=False)             # 隨機選一個 sample
                    selected_indices.extend(indices)
                    ds_info[random_label].remove(indices[0])

                random.shuffle(selected_indices)
                selected_dataset = Subset(datasets[dataset_name], selected_indices)
                client_datasets[c][dataset_name] = selected_dataset

        for dataset_name in df_partition:
            df = pd.DataFrame(df_partition[dataset_name])
            df.index = ["client_" + str(i) for i in range(num_clients)]
            df.columns = ["label_" + str(i) for i in range(num_classes)]
            dash = "-" * (df.shape[1] * 4)
            print("\n{} [{}] partition: {}\n{}".format(dash, dataset_name, dash, df))
            # print("\n{} [{}] statistics: {}\n{}".format(dash, name, dash, pd.DataFrame(df.sum()).T))

        print("\n######### Client datasets #########")
        df2 = pd.DataFrame({name: {c: len(client_datasets[c][name]) for c in client_datasets} for name in DataSet},)
        df2.index = ["client_" + str(i) for i in range(num_clients)]
        print(df2)
        print("{}\n".format("#" * 35))

        return client_datasets, balanced_testset


    ''' Partition dataset by class into different tasks for class-incremental learning '''
    def divide_dataset_to_incremental_partition(self, dataset, args):
        num_classes = args.num_classes
        task_labels = np.array_split(np.arange(num_classes), 5)

        ''' Get each label's dataset index '''
        info = {i: [] for i in range(num_classes)}
        for idx, (img, label) in enumerate(dataset):
            info[label].append(idx)

        ''' Divide the dataset into 5 tasks (assign each task with 2 classes)'''
        task_partition = {
            f"task_{i+1}": np.concatenate([info[label] for label in task_labels[i]])
            for i in range(5)
        }

        task_dataset = {
            "task_1": Subset(dataset, task_partition["task_1"]),
            "task_2": Subset(dataset, task_partition["task_2"]),
            "task_3": Subset(dataset, task_partition["task_3"]),
            "task_4": Subset(dataset, task_partition["task_4"]),
            "task_5": Subset(dataset, task_partition["task_5"]),
        }
        return task_dataset


    # ------------------------
    # IID Partition for Clients
    # ------------------------
    def iid_partition(self, dataset, args):
        random.seed(args.seed+2)
        np.random.seed(args.seed+2)
        num_clients = args.num_clients
        

        Dataset, Testset = self.load_data(dataset_name=dataset, args=args)
        data_len = len(Dataset)
        indices = np.random.permutation(data_len)
        client_size = data_len // num_clients

        client_datasets = {i: {} for i in range(num_clients)}
        testset = {dataset: Testset}

        for i in range(num_clients):
            start = i * client_size
            end = data_len if i == num_clients - 1 else (i + 1) * client_size
            client_indices = indices[start:end]
            client_datasets[i][dataset] = Subset(Dataset, client_indices)

        return client_datasets, testset



def load_public_data(dataset, args: ArgumentParser):
    if dataset == "Mnist":
        unlabled_dataset, _ = DataModel().load_data(dataset_name="EMNIST", args=args)
    elif dataset == "Cifar10":
        unlabled_dataset, _ = DataModel().load_data(dataset_name="CIFAR100", args=args)
    elif dataset == "Office-Caltech":
        unlabled_dataset, _ = DataModel().load_data(dataset_name="Caltech101", args=args)
    else:
        raise ValueError("No supported unlabeled dataset for : {}".format(dataset))
    return unlabled_dataset


# if __name__ == "__main__":
#     load_noniid_partition(args)