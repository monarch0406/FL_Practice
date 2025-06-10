import random
import numpy as np
from dpmodels.dpmodel import BaseDPModel



class incremental_arrival(BaseDPModel):
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

        ''' There are `inital_clients` in the current environment. 
            The rest of the clients will arrive incrementally from the `round_start` round. 
            
            For example, if there are 3 clients and 6 rounds, the client_state matrix will be:
            ------------------------------------
            |  Client  |         Round         |
            ------------------------------------
            | client 0 | 1 | 1 | 1 | 1 | 1 | 1 |
            | client 1 | 0 | 0 | 1 | 1 | 1 | 1 |
            | client 2 | 0 | 0 | 0 | 0 | 1 | 1 |
        '''
        
        initial_clients = self.args.initial_clients

        # let the `initial_clients` clients be the same as the 'non-important' case
        # (most important client -> least important client)
        # if self.isimportant:
        #     self.client_id = self.sort_client_by_importance(starts_from=initial_clients)        
        #     self.client_id[initial_clients:] = self.client_id[initial_clients:][::-1]
        self.client_id = np.arange(self.num_clients)

        ''' The `initial_clients` clients will participate in all rounds. '''
        for c in range(initial_clients):
            self.client_state[self.client_id[c], :] = 1

        ''' The rest of the clients will arrive incrementally from the `round_start` round.'''
        i = self.round_start - 1
        # i = 50
        
        for c in range(initial_clients, self.num_clients):
            self.client_state[self.client_id[c], i:] = 1
            i += self.interval

            if i >= self.num_rounds:
                break

        return self.client_state