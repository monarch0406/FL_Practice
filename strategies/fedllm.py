from typing import List, Tuple, Dict
from colorama import Fore, Style
from collections import deque
from .fedavg import FedAvg
import numpy as np
import re
import copy
import json
import torch
import torch.nn.functional as F
from functools import reduce
from utils.util import set_param
from openai import OpenAI

# openai.api_key = os.getenv("OPENAI_API_KEY")  # or paste your key directly

np.set_printoptions(suppress=True, linewidth=np.inf)

'''
Hyperparameters:
    - dataset: MNIST, CIFAR-10, Office-Caltech
'''

class FedLLM(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        # self.fl_strategy = fl_strategy
        
        ''' Parameters for client data statistics '''
        self.num_classes = args.num_classes
        self.label_weights = []

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
        self.history_log = {}
        self.history_length = 5  # set your desired history length


    def _initialization(self, **kwargs) -> None:
        self.SYSTEM_PROMPT = """
            You are an intelligent federated learning orchestrator. Your role is to assign aggregation weights to client models in a dynamic setting, where clients may join or leave at any round.
            ---

            🎯 Objective:
            Your primary goal is to **maximize the generalization performance of the global model** over time.

            ---

            📊 Evaluation Guidelines:
            You must consider the following for each client:

            1. **Participation Pattern**  
            Use `participate_history` and `current_round` to identify activity.  
            - A stale client ≠ useless. A previously consistent but now inactive client may still contain essential knowledge.

            2. **Generalization Quality**  
            Use `global_eval_scores` to assess global performance impact.

            3. **Class Coverage & Bias**  
            Use `class_accuracy` and `data_distribution` to check if the client:
            - Overfits to majority local classes
            - Provides rare or underrepresented class data

            ---

            ⚖️ Important:
            Do **not** penalize stale clients *just* for being inactive. Instead:
            - Check if they hold rare knowledge
            - Check if they were historically strong contributors
            - Consider their complementary value to the rest of the pool

            ---

            📥 Input Format:
            You will receive a JSON list in the following format:
            ```json
            {
            "current_round": 10,
            "history_log": { 
                    "round_0": {
                        "weight_assignments": {"client_id": "client_01", "weight": 0.23, "reason": "Explain why this client received this weight."}
                        "strategy_summary": "Prioritized clients with large, balanced datasets to encourage initial generalization."
                    },
                    ...
            },
            "clients": [
                {
                "client_id": "client_03",
                "participate_history": [1, 2, 4, 6],
                "global_eval_scores": [0.75, 0.77],
                "class_accuracy": {"0": 0.92, "1": 0.63, "2": 0.20},
                "data_distribution": {"0": 0.6, "1": 0.2, "2": 0.2}
                },
                ...
            ]
            }

            📤 Output Format (STRICT):
            You should return a **JSON object** with two fields:

            {
            "weights": [
                    {
                    "client_id": "client_01",
                    "weight": 0.23,
                    "reason": "Explain why this client received this weight."
                    },
                    ...
                ]
            }

            Constraints:
            - The sum of weights should be approximately 1.0
            - Each client must have a clear, strategy-aligned justification
            - Reuse or adapt your previous strategies as needed, especially when feedback on performance is available

            ---

            📈 Hint:
            If your strategy underperforms over time (e.g., poor global model improvement), **revise it** in the next round.
        """
    

    def build_user_prompt(self, client_info_list):
        return f"Here is the current client metadata:\n{json.dumps(client_info_list, indent=2)}"


    def query_openai_contribution_weights(self, client_info_list):
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.build_user_prompt(client_info_list)}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content


    def generate_prompt_ready_metadata(self, rounds):
        """
        Convert self.knowledge_pool into LLM-ready JSON input.
        Ensures required keys are included and sorted for readability.
        """
        prompt_ready = [f"Current round: {rounds}"]
        # prompt_ready.append(f"history_log: {self.history_log}")

        for sig, item in self.knowledge_pool.items():
            entry = {
                "client_id": item["client_id"],
                "active_rounds": item.get("active_round", []),
                "dataset_size": item.get("dataset_size", 0),
                "data_distribution": item.get("data_distribution", {}),
                "global_eval_scores": list(item.get("global_eval_scores", [])),
                "class_accuracy": item.get("class_accuracy", {}),
                # "gradient_alignment": item.get("gradient_alignment", 0.0),
            }
            prompt_ready.append(entry)

        return prompt_ready


    def update_feedback(self, rounds: int, response: str):
        self.history_log[f"round_{rounds}"] = {
            # "global_accuracy": global_accuracy,
            "weight_assignments": response,
            # "strategy_summary": strategy
        }


    def parse_llm_output(self, response_text):
        """
        Parse the LLM response to extract client weights and reasons.
        Ensures the output is a valid JSON list and handles any parsing errors.
        """
        # print("LLM response:", response_text)
        # input()

        try:
            json_str = re.search(r"\{.*\}", response_text, re.DOTALL).group()
            result = json.loads(json_str)
            # print(result)
            
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
        
        # Normalize weights
        total = sum([r["weight"] for r in result["weights"]])
        for r in result["weights"]:
            r["weight"] = round(r["weight"] / total, 6)  # normalize

        weights = [r["weight"] for r in result["weights"]]
        return result, weights


    # Print results
    def print_results(self, response_text):
        print("\n" + "-" * 150)
        # print(Fore.RED + "[Strategy]:\n" + response_text["strategy"] + Fore.RESET)
        for client in response_text["weights"]:
            print(f"[{client['client_id']}]: weight: " + Fore.GREEN + f"{client['weight']:>5.3f}" + Fore.RESET + f" | Reason: " + Fore.YELLOW + f"{client['reason']}" + Fore.RESET)
        print("-" * 150)


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        # prev_global_model = copy.deepcopy(global_model)

        # 1. Aggregate the active client's models
        # new_weights = self._aggregation(client_list)
        # set_param(global_model, new_weights)

        # grad_global = self.flatten(global_model) - self.flatten(prev_global_model)
        self.update_knowledge_pool(client_list, active_clients, rounds)

        # 2. Generate the prompt-ready metadata
        client_info_list = self.generate_prompt_ready_metadata(rounds)
        print("Query:\n", client_info_list)
        # input()

        # Pass it to LLM query
        response_text = self.query_openai_contribution_weights(client_info_list)
        text, agg_weights = self.parse_llm_output(response_text)
        self.print_results(text)
        # input()

        self.update_feedback(rounds, text["weights"])

        self.update_weight_in_knowledge_pool(text, agg_weights)
        self.show_knowledge_pool()
        # input()

        # 3. Aggregate the models in the knowledge pool
        new_weights = self.aggregate_from_knowledge_pool(agg_weights)
        return new_weights


    def update_knowledge_pool(self, client_list, active_clients, rounds, global_model=None, grad_global=None):
        for cid in active_clients:
            client = client_list[cid]

            result = client.test()
            test_acc, class_acc = result["test_acc"], result["class_acc"]
            # g_i = self.gradient_alignment(client.model, global_model, grad_global)

            # signature = "{cid}-{data}".format(cid=client.cid, data=str(list(client.data_distribution.keys())))
            signature = "client_{:02d}".format(client.cid)
            # print(client.data_distribution)

            if signature in self.knowledge_pool:                                    # update the snapshot in knowledge pool
                item = self.knowledge_pool[signature]
                item["model"] = copy.deepcopy(client.model)
                item["active_round"].append(rounds)
                item["dataset_size"] = len(client.trainLoader.dataset)
                item["data_distribution"] = client.data_distribution
                item["global_eval_scores"].append(test_acc)
                item["class_accuracy"] = class_acc
                # item["gradient_alignment"] = g_i
            else:                                                               # add a new snapshot to knowledge pool
                self.knowledge_pool[signature] = {
                        "client_id": "client_{:02d}".format(client.cid),
                        "model": copy.deepcopy(client.model),
                        "active_round": [rounds],
                        "dataset_size": len(client.trainLoader.dataset),
                        "data_distribution": client.data_distribution,
                        "global_eval_scores": deque(maxlen=self.history_length),
                        "class_accuracy": class_acc,
                        # "gradient_alignment": g_i,
                        "weight": 0.0,
                    }
                self.knowledge_pool[signature]["global_eval_scores"].append(test_acc)


    def update_weight_in_knowledge_pool(self, result, weights):
        """
        Update the weight of a specific client in the knowledge pool.
        """
        for client, agg_weight in zip(result["weights"], weights):
            cid, reason = client["client_id"], client["reason"]
            self.knowledge_pool[cid]["prev_weight"] = self.knowledge_pool[cid]["weight"]
            self.knowledge_pool[cid]["weight"] = agg_weight
            self.knowledge_pool[cid]["reason"] = reason


    def show_knowledge_pool(self,):
        print("\n-----> Knowledge Pool Status, Size: {}".format(len(self.knowledge_pool)))
        for key, item in self.knowledge_pool.items():
            line_head = "[{}]".format(item["client_id"])
            line_mid1 = "Acc.: {}{:}".format(Fore.RED, [round(x, 2) for x in item["global_eval_scores"]]) + Fore.RESET
            # line_mid2 = "Grad: {}{}".format(Fore.MAGENTA, item["gradient_alignment"]) + Fore.RESET
            line_mid2 = "Weight: {}{:.2f} -> {:.2f}".format(Fore.GREEN, item["prev_weight"], item["weight"]) + Fore.RESET
            line_tail = "Reason: {}{}".format(Fore.YELLOW, item["reason"]) + Fore.RESET
            print("{:<12} {:45} | {:<20} | {:<100}".format(line_head, line_mid1, line_mid2, line_tail))

            # line_mid2 = "{}On. Age: {}".format(Fore.GREEN if value["online_age"] > 0 else Fore.LIGHTBLACK_EX, value["online_age"]) + Fore.RESET
            # line_tail = "{}Off. Age: {}".format(Fore.RED if value["offline_age"] > 0 else Fore.LIGHTBLACK_EX, value["offline_age"]) + Fore.RESET
            # print("{:<55} {:<25} {:<26} {:<27}".format(line_head, line_mid1, line_mid2, line_tail))
        print("-" * 101 + "\n")


    def aggregate_from_knowledge_pool(self, agg_weight) -> np.ndarray:
        ''' Aggregate with all models in the knowledge pool (no selection) '''
        model_weights = [
            [value["model"].state_dict()[params] for params in value["model"].state_dict()] for sig, value in self.knowledge_pool.items()
        ]

        # print("Agg_weights:", agg_weight)
        # input()

        # Compute average weight of each layer using the findal weights
        weights_prime = [
            reduce(torch.add, [w * weight for w, weight in zip(layer_updates, agg_weight)])
            for layer_updates in zip(*model_weights)
        ]
        return weights_prime
    

    # def evaluate(self, server_model, testLoader) -> Tuple[float, float]:
    #     avg_acc = []
    #     for name, loader in testLoader.items():
    #         # _, acc_indi, _ = self.fl_strategy._test(server_model, loader)
    #         acc_indi = self.fl_strategy._test(server_model, loader)["test_acc"]
    #         avg_acc.append(acc_indi)
    #     return np.mean(avg_acc)
    

    def flatten(self, model):
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def gradient_alignment(self, client_model, global_model, reference_update):
        g_i = self.flatten(client_model) - self.flatten(global_model)
        return F.cosine_similarity(g_i, reference_update, dim=0).item()