import random
import numpy as np
from dpmodels.dpmodel import BaseDPModel



class round_robin(BaseDPModel):
    def __init__(self, args):
        super().__init__(args)

        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        self.round_start = args.round_start
        self.interval = args.interval
        self.overlap_clients = args.overlap_clients

        self.client_state = np.zeros((self.num_clients, self.num_rounds))
        self.client_state = self.set_pattern()

    
    ''' Schedule clients for each round '''
    def set_pattern(self,):
        random.seed(self.args.seed+2)
        np.random.seed(self.args.seed+2)

        ''' There are `num_clients` clients need to be iterated participating with `interval` rounds.
            Clients may 
            For example, if there are 3 clients and 6 rounds, the client_state matrix will be:
            ------------------------------------
            |  Client  |         Round         |
            ------------------------------------
            | client 0 | 1 | 1 | 0 | 0 | 0 | 0 |
            | client 1 | 1 | 1 | 1 | 1 | 0 | 0 |
            | client 2 | 0 | 0 | 1 | 1 | 1 | 1 |
            | client 3 | 0 | 0 | 0 | 0 | 1 | 1 |
        '''

        i = self.round_start - 1
        while i < self.num_rounds:
            for c in range(self.num_clients):
                for k in range(self.overlap_clients):
                    self.client_state[(c + k) % self.num_clients, i: min(i+self.interval, self.num_rounds)] = 1

                i += self.interval
                if i >= self.num_rounds:
                    break

        return self.client_state