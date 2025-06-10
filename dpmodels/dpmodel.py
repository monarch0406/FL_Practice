from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

'''
    This is the implementation of the client scheduler. There are 5 types of client dynamics:
    - static: all clients participate in all rounds
    - round-robin: clients participate in a round-robin fashion
    - incremental_arrival: clients arrive incrementally from a specific round
    - incremental_departure: clients depart incrementally from a specific round
    - random: clients participate randomly in each round        
    - markov: clients participate based on a Markov chain

    - client_state: a matrix to store the state of clients in each round
    ------------------------------------
    |  Client  |         Round         |
    ------------------------------------
    | client 0 | 1 | 1 | 0 | 0 | 0 | 0 |
    | client 1 | 0 | 0 | 1 | 1 | 0 | 0 |
    | client 2 | 0 | 0 | 0 | 0 | 1 | 1 |
'''

# parser = argparse.ArgumentParser("FL Dynamic-Client-Participation")
# parser.add_argument("--dynamic_type", default="round-robin", type=str)
# parser.add_argument("--important_client", action="store_true", help="sampled by important client or not")
# parser.add_argument("--num_clients", default=5, type=int)
# parser.add_argument("--num_rounds", default=20, type=int)
# parser.add_argument("--round_start", default=5, type=int)
# parser.add_argument("--initial_clients", default=2, type=int)
# parser.add_argument("--interval", default=3, type=int, help="interval for incremental arrival dynamic")
# parser.add_argument("--seed", default=42, type=int)
# args = parser.parse_args()


class BaseDPModel(ABC):
    """
    Abstract base class for client participation scheduling in federated learning.
    Subclass this and implement `set_pattern()` to define your participation logic.

    Attributes:
        args: Namespace or dict containing configuration (must include num_clients, num_rounds, etc.)
        client_state: np.ndarray of shape (num_clients, num_rounds), 1 if client participates in round, else 0

    Example:
        class MyPattern(BaseDPModel):
            def set_pattern(self):
                # Custom logic to fill self.client_state
                ...
                return self.client_state
    """

    def __init__(self, args):
        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        self.args = args
        self.client_state = np.zeros((self.num_clients, self.num_rounds), dtype=int)        # build a matrix to store the state of clients in each round
        # self.client_state = self.set_pattern()
        # self.client_id = self.sort_client_by_importance()                     # build a list of client ids
        # self.client_id = np.arange(self.num_clients)

    ''' Schedule clients for each round '''
    @abstractmethod
    def set_pattern(self):
        """
        Define the participation pattern by filling self.client_state.
        Must return the participation matrix (np.ndarray).
        """

        # random.seed(self.args.seed+2)
        # np.random.seed(self.args.seed+2)
        raise NotImplementedError("set_pattern() must be implemented in your strategy.")
        

    ''' Update the clients for the current round '''
    def update(self, round):
        """
        Get active and inactive client indices for a given round (1-based).
        Returns:
            active_ids: np.ndarray of active client indices
            inactive_ids: np.ndarray of inactive client indices
        """
        current_state = self.client_state[:, round-1]

        active_ids = np.where(current_state == 1)[0]
        inactives_ids = np.where(current_state == 0)[0]
        return active_ids, inactives_ids

    
    def visualize(self):
        """
        Optional: Visualize the participation matrix.
        """
        plt.imshow(self.client_state, aspect='auto', cmap='Greys')
        plt.xlabel("Round")
        plt.ylabel("Client")
        plt.title("Client Participation Matrix")
        plt.show()

    # def sort_client_by_importance(self, starts_from=0):
    #     if self.isimportant:
    #         client_ids = [i for i in range(starts_from)]
    #         result = []
    #         for i in range(starts_from, self.num_clients):
    #             num_data = len(self.client_list[i].trainLoader.dataset)
    #             result.append([self.client_list[i].cid, num_data])
    #             # print(f"Client {c.cid}: {num_data} data")

    #         ''' sort the clients based on the number of data '''
    #         result.sort(key=lambda x: x[1], reverse=False)
    #         print("Important client: {}\n".format(result))

    #         client_ids += [x[0] for x in result]
    #         return client_ids
    #     else:
    #         return np.arange(self.num_clients)


# scheduler = ClientScheduler(args)
# print(scheduler.schedule())

# for r in range(20):
#     active_ids, inactives_ids = scheduler.update(r)
#     print(f"Round {r}: active clients: {active_ids}, inactive clients: {inactives_ids}")