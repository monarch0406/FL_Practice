import random
import numpy as np
from dpmodels.dpmodel import BaseDPModel



class incremental_departure(BaseDPModel):
    def __init__(self, args):
        super().__init__(args)

        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        self.round_start = args.round_start
        self.interval = args.interval

        self.client_state = np.zeros((self.num_clients, self.num_rounds))
        self.client_state = self.set_pattern()

    
    ''' Schedule clients for each round '''
    def set_pattern(self,):
        random.seed(self.args.seed+2)
        np.random.seed(self.args.seed+2)

        ''' All clients are in the current environment.
            The clients will depart incrementally from the `round_start` round.

            For example, if there are 3 clients and 6 rounds, the client_state matrix will be:
            ------------------------------------
            |  Client  |         Round         |
            ------------------------------------
            | client 0 | 1 | 1 | 1 | 1 | 1 | 1 |
            | client 1 | 1 | 1 | 1 | 1 | 0 | 0 |
            | client 2 | 1 | 1 | 0 | 0 | 0 | 0 |
        '''

        initial_clients = self.args.initial_clients

        ''' All clients are in the current environment. '''
        self.client_state[:, :] = 1

        ''' The clients will depart incrementally from the `round_start` round. '''
        i = self.round_start - 1
        # i = 50

        # (least important client -> most important client)
        # if self.isimportant:
        #     self.client_id[initial_clients:] = self.client_id[initial_clients:][::-1]
        self.client_id = np.arange(self.num_clients)

        for c in range(self.num_clients, initial_clients, -1):
            self.client_state[self.client_id[c-1], i:] = 0
            i += self.interval

            if i >= self.num_rounds:
                break

        return self.client_state