from typing import List, Tuple
from .fedavg import FedAvg
import numpy as np
import time
from datetime import datetime
import nni
import os
import shutil
import copy
import torch
from functools import reduce
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from models.model_target import Generator, GlobalSynthesizer, KLDiv, Normalizer, UnlabeledImageDataset, weight_init
from utils.util import ConcatDataset, set_param

'''
Hyperparameters (From paper):
    - temperature: 2
'''

class Target(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.student_model = None
        self.syn_data_loader = None
        self.syn_iter = None
        self.alpha = 20
        self.T = 7                            # temperature-scaled logit for distillation loss
        self.gen_T = 15

        self.synthesis_batch_size = 128
        self.sample_batch_size = 128
        self._total_classes = 10
        self.nums = 1000
        self.g_steps = 20
        self.is_maml=1
        self.kd_steps = 40
        self.warmup = 8
        self.lr_g=0.002
        self.lr_z=0.01
        self.oh = 3
        self.bn = 13
        self.adv = 2
        self.act=0.0
        self.reset_l0=1
        self.reset_bn=0
        self.bn_mmt=0.9
        self.syn_round = 25
        self.tau=1

        self.data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**dict(self.data_normalize)),
        ])

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")
        self.save_dir = "./data/synthetic_data/{}/{}/{}/".format(self.args.dynamic_type, self.args.seed, timestamp)
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)


    def _initialization(self, **kwargs) -> None:
        client_list = kwargs["client_list"]

        # self.strategy.generator = Generator(nz=256, ngf=64, nc=3, img_size=32).to(self.device)
        # self.strategy.student_model = copy.deepcopy(self.model)
        self.syn_data_loader = None

        ''' Get client's validation data '''
        valset = [client_list[i].valLoader.dataset for i in range(len(client_list))]
        self.valLoader = {
            "total_val": DataLoader(ConcatDataset(valset), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        }


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        global_model = kwargs["global_model"]

        # train_loss, train_acc, dist_loss = self.client_list[i].train(rounds=rounds, syn_data_loader=self.strategy.syn_data_loader, prev_global_model=self.strategy.prev_global_model, task_id=self._current_tid)
        result = client_list[cid].train(rounds=rounds, syn_data_loader=self.syn_data_loader, prev_global_model=global_model, task_id=rounds)
        train_loss, train_acc, dist_loss = result["train_loss"], result["train_acc"], result["dist_loss"]
        print("[Client {}] loss: {:.4f}, dist_loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, dist_loss, train_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)

        set_param(global_model, new_weights)                                    # set global model's weight for distillation
        self.data_generation(global_model=global_model, rounds=rounds)
        self.syn_data_loader = self.get_syn_data_loader(rounds=rounds)
        self.syn_iter = iter(self.syn_data_loader)
        # pass
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
        # global_model = copy.deepcopy(model).to(self.device)
        # global_model.eval()

        syn_data_loader, prev_global_model, task_id = kwargs["syn_data_loader"], kwargs["prev_global_model"], kwargs["task_id"]

        prev_global = copy.deepcopy(prev_global_model).to(self.device)
        prev_global.eval()

        model.train()
        for i in range(num_epochs):
            total_loss, total_correct = [], []
            dist_loss = []
            
            for idx, (x, label)  in enumerate(trainLoader):
                x, label = x.to(self.device), label.to(self.device)
                logit_student, _ = model(x)

                predict = torch.argmax(logit_student.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))

                loss = F.cross_entropy(logit_student, label)

                ''' Distillation from global model using synthesized data '''
                if task_id > 1:
                    try:
                        syn_input = next(self.syn_iter)
                    except:
                        self.syn_iter = iter(syn_data_loader)
                        syn_input = next(self.syn_iter)
                    syn_input = syn_input.to(self.device)

                    logit_syn, _ = model(syn_input)
                    with torch.no_grad():
                        logit_teacher, _ = prev_global(syn_input.detach())

                    kd_loss = self.KD_loss(logit_syn, logit_teacher.detach())
                    dist_loss.append(kd_loss.item())

                    loss += self.alpha * kd_loss
                else:
                    dist_loss = [0]

                total_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)               # clip the gradient to prevent exploding
                optimizer.step()

        # return np.mean(total_loss), np.mean(total_correct), np.mean(dist_loss)
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct),
            "dist_loss": np.mean(dist_loss)
        }


    def data_generation(self, global_model, rounds):
        global_model = copy.deepcopy(global_model).to(self.device)
        global_model.eval()

        nz = 256
        img_size = 32
        img_shape = (3, 32, 32)
        normalizer = Normalizer(**dict(self.data_normalize))

        tmp_dir = os.path.join(self.save_dir, "task_{}".format(rounds))
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True) 

        # generator = self.generator
        # student = self.student_model

        student = copy.deepcopy(global_model).to(self.device)
        student.apply(weight_init)
        generator = Generator(nz=nz, ngf=64, nc=3, img_size=img_size).to(self.device)
        synthesizer = GlobalSynthesizer(copy.deepcopy(global_model), student, generator,
                    nz=nz, num_classes=self._total_classes, img_size=img_shape, init_dataset=None,
                    save_dir=tmp_dir,
                    transform=self.train_transform, normalizer=normalizer,
                    synthesis_batch_size=self.synthesis_batch_size, sample_batch_size=self.sample_batch_size,
                    iterations=self.g_steps, warmup=self.warmup, lr_g=self.lr_g, lr_z=self.lr_z,
                    adv=self.adv, bn=self.bn, oh=self.oh,
                    reset_l0=self.reset_l0, reset_bn=self.reset_bn,
                    bn_mmt=self.bn_mmt, is_maml=self.is_maml, args=self.args)
        
        criterion = KLDiv(T=self.gen_T)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.syn_round, eta_min=2e-4)

        for it in range(self.syn_round):
            synthesizer.synthesize()                                                    # generate synthetic data
            if it >= self.warmup:
                self.kd_train(student, global_model, criterion, optimizer, rounds)      # kd_steps
                # _, test_acc, _ = self._test(student, self.valLoader['total_val'])
                test_acc = self._test(student, self.valLoader['total_val'])["test_acc"]

                print("Task {}, Data Generation, Epoch {}/{} =>  Student test_acc: {:.2f}".format(
                    rounds, it + 1, self.syn_round, test_acc,))
                scheduler.step()
                # wandb.log({'Distill {}, accuracy'.format(self._cur_task): test_acc})

        del synthesizer
        print("For task {}, data generation completed! ".format(rounds))  


    def kd_train(self, student, teacher, criterion, optimizer, rounds):
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data(rounds) 
        data_iter = iter(loader)
    
        for i in range(self.kd_steps):
            try:
                images = next(data_iter).cuda()
            except:
                data_iter = iter(loader)
                images = next(data_iter).cuda()

            with torch.no_grad():
                t_out, _ = teacher(images)
            s_out, _ = student(images)
            loss_s = criterion(s_out, t_out)
            optimizer.zero_grad()
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
            optimizer.step()


    def get_syn_data_loader(self, rounds):
        data_dir = os.path.join(self.save_dir, "task_{}".format(rounds))
        # print("data_dir: {}".format(data_dir))

        syn_dataset = UnlabeledImageDataset(data_dir, transform=self.train_transform, nums=self.nums)
        syn_data_loader = torch.utils.data.DataLoader(syn_dataset, batch_size=self.sample_batch_size, shuffle=True ,num_workers=2)

        # public_subset = torch.utils.data.Subset(self.valset, np.random.choice(len(self.valset), 1000, replace=False))
        # syn_data_loader = torch.utils.data.DataLoader(public_subset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        return syn_data_loader


    def get_all_syn_data(self, rounds):
        data_dir = os.path.join(self.save_dir, "task_{}".format(rounds))
        syn_dataset = UnlabeledImageDataset(data_dir, transform=self.train_transform, nums=1000)
        loader = torch.utils.data.DataLoader(syn_dataset, batch_size=self.sample_batch_size, shuffle=True, sampler=None)
        return loader


    def _test(self, model, testLoader) -> Tuple[float, float]:
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


    def KD_loss(self, pred, soft):
        pred = torch.log_softmax(pred / self.T, dim=1)
        soft = torch.softmax(soft / self.T, dim=1)
        return 1 * torch.mul(soft, pred).sum() / pred.shape[0]
    
    def KL_loss(self, student_logits, teacher_logits):
        divergence = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction="batchmean",
        )
        return divergence