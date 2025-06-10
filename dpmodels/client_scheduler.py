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


class DPModel:
    def __init__(self, client_list, args):
        self.client_list = client_list
        self.num_clients = args.num_clients
        self.overlap_clients = args.overlap_clients
        self.num_rounds = args.num_rounds
        self.round_start = args.round_start
        self.dynamic_type = args.dynamic_type
        self.interval = args.interval
        self.isimportant = args.important_client
        self.args = args

        self.client_id = self.sort_client_by_importance()                     # build a list of client ids
        self.client_state = np.zeros((self.num_clients, self.num_rounds))     # build a matrix to store the state of clients in each round
        self.client_state = self.schedule()

    ''' Schedule clients for each round '''
    def schedule(self):
        random.seed(self.args.seed+2)
        np.random.seed(self.args.seed+2)

        if self.dynamic_type == "static":
            self.client_state[:, :] = 1

        elif self.dynamic_type == "round-robin":
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

        elif self.dynamic_type == "incremental-arrival":
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
            if self.isimportant:
                self.client_id = self.sort_client_by_importance(starts_from=initial_clients)        
                self.client_id[initial_clients:] = self.client_id[initial_clients:][::-1]

            ''' The `initial_clients` clients will participate in all rounds. '''
            for c in range(initial_clients):
                self.client_state[self.client_id[c], :] = 1

            ''' The rest of the clients will arrive incrementally from the `round_start` round.'''
            # i = self.round_start - 1
            i = 50
            for c in range(initial_clients, self.num_clients):
                self.client_state[self.client_id[c], i:] = 1
                i += self.interval

                if i >= self.num_rounds:
                    break
            
        elif self.dynamic_type == "incremental-departure":
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
            # i = self.round_start - 1
            i = 50

            # (least important client -> most important client)
            if self.isimportant:
                self.client_id[initial_clients:] = self.client_id[initial_clients:][::-1]

            for c in range(self.num_clients, initial_clients, -1):
                self.client_state[self.client_id[c-1], i:] = 0
                i += self.interval

                if i >= self.num_rounds:
                    break

        elif self.dynamic_type == "random":
            ''' Randomly assign clients to participate in each round. 
                For example, if there are 3 clients and 6 rounds, the client_state matrix will be:
                ------------------------------------
                |  Client  |         Round         |
                ------------------------------------
                | client 0 | 1 | 0 | 1 | 0 | 0 | 1 |
                | client 1 | 0 | 1 | 0 | 1 | 1 | 0 |
                | client 2 | 0 | 0 | 0 | 0 | 1 | 1 |
            '''

            if self.isimportant:
                top_k = int(self.num_clients * self.args.top_ratio)
                important_clients = self.client_id[-top_k:]
                # print("Important clients: ", important_clients)

                for c in range(self.num_clients):
                    if c in important_clients:
                        self.client_state[c, :] = np.random.randint(0, 2, self.num_rounds)
                    else:
                        self.client_state[c, :] = 1
            else:
                for c in range(self.num_clients):
                    self.client_state[c, :] = np.random.randint(0, 2, self.num_rounds)

                # check if there is at least one client participating in each round
                for r in range(self.num_rounds):
                    if np.sum(self.client_state[:, r]) == 0:
                        c = np.random.randint(0, self.num_clients)
                        self.client_state[c, r] = 1

        elif self.dynamic_type == "markov":
            ''' Clients participate based on a Markov chain.
                There are 2 states: active and inactive.
                - Active: `active_prob` probability to stay active in the next round, `1-active_prob` probability to become inactive.
                - Inactive: `inactive_prob` probability to stay inactive in the next round, `1-inactive_prob` probability to become active.

                  (1 - active_prob)
                      --------
                      |      |
                      v      |  
                ------------  ==> active_prob ===>  ----------
                | inactive |                        | active |
                ------------  <== inactive_prob ==  ----------
                                                 |      ^
                                                 |      |
                                                 --------
                                            (1 - inactive_prob)

            '''

            transition_matrix = np.array([
                [0.2, 0.8],                     # inactive -> active, inactive -> inactive
                [0.8, 0.2],                     #   active -> active,   active -> inactive
            ])

            
            if self.isimportant:
                top_k = int(self.num_clients * self.args.top_ratio)
                important_clients = self.client_id[-top_k:]
                # print("Important clients: ", important_clients)

                ''' Randomly assign the initial state (round=0) for each client '''
                for c in range(self.num_clients):
                    if c in important_clients:
                        self.client_state[c, 0] = np.random.choice([0, 1])
                    else:
                        self.client_state[c, 0] = 1

                ''' Update the state of clients based on the transition matrix '''
                for r in range(1, self.num_rounds):
                    for c in range(self.num_clients):
                        if c in important_clients:
                            self.client_state[c, r] = np.random.choice([1, 0], p=transition_matrix[int(self.client_state[c, r-1])])
                        else:
                            self.client_state[c, r] = 1
            else:
                ''' Randomly assign the initial state (round=0) for each client '''
                self.client_state[:, 0] = np.random.choice([0, 1], self.num_clients)

                ''' Update the state of clients based on the transition matrix '''
                for r in range(1, self.num_rounds):
                    for c in range(self.num_clients):
                        self.client_state[c, r] = np.random.choice([1, 0], p=transition_matrix[int(self.client_state[c, r-1])])

                # check if there is at least one client participating in each round
                for r in range(self.num_rounds):
                    if np.sum(self.client_state[:, r]) == 0:
                        c = np.random.randint(0, self.num_clients)
                        self.client_state[c, r] = 1

        return self.client_state

    
    def sort_client_by_importance(self, starts_from=0):
        if self.isimportant:
            client_ids = [i for i in range(starts_from)]
            result = []
            for i in range(starts_from, self.num_clients):
                num_data = len(self.client_list[i].trainLoader.dataset)
                result.append([self.client_list[i].cid, num_data])
                # print(f"Client {c.cid}: {num_data} data")

            ''' sort the clients based on the number of data '''
            result.sort(key=lambda x: x[1], reverse=False)
            print("Important client: {}\n".format(result))

            client_ids += [x[0] for x in result]
            return client_ids
        else:
            return np.arange(self.num_clients)


    ''' Update the clients for the current round '''
    def update(self, round):
        current_state = self.client_state[:, round-1]

        active_ids = np.where(current_state == 1)[0]
        inactives_ids = np.where(current_state == 0)[0]
        return active_ids, inactives_ids


# scheduler = ClientScheduler(args)
# print(scheduler.schedule())

# for r in range(20):
#     active_ids, inactives_ids = scheduler.update(r)
#     print(f"Round {r}: active clients: {active_ids}, inactive clients: {inactives_ids}")