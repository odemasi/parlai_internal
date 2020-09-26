

from parlai.core.teachers import FixedDialogTeacher
# from .build import build
import os
import json

import random
import sys

    



def _path(opt):
    # build the data if it does not exist
#     build(opt)

    # set up path to data (specific to each dataset)
    print('DATAPATH:', opt['datapath'])
    jsons_path = os.path.join(opt['datapath'], 'multiwoz_v21', 'MULTIWOZ2.1')
    conversations_path = os.path.join(jsons_path, 'data.json')
    return conversations_path, jsons_path


class FtmlTeacher(FixedDialogTeacher):
    """
    Copied from MultiWOZ Teacher.
    This dataset contains more than just dialogue. It also contains:
    data.json also contains the following information:
    1. information about the goal of the conversation (what the person is looking for)
    2. topic of the conversation
    3. log of the conversation (the conversation itself + metadata about each utterance)
          Metadata: any taxi, police, restaurant, hospital, hotel, attraction, or train info mentioned
    Information about each metadata category is also contained in its own json file.
    1. restaurant.json: Restaurants + their attributes in the Cambridge UK area (address, pricerange, food...)
    2. attraction.json: Attractions + their attributes in the Cambridge UK area (location, address, entrance fee...)
    3. hotel_db.json: Hotels + their attributes in the Cambridge UK area  (address, price, name, stars...)
    4. train_db.json: Trains + their attributes in the Cambridge UK area (destination, price, departure...)
    5. hospital_db.json: data about the Cambridge hospital's departments (department, phone, id)
    6. police_db.json: Name address and phone number of Cambridge police station
    7. taxi_db.json: Taxi information (taxi_colors, taxi_types, taxi_phone)
    More information about the jsons can be found in readme.json
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt['datafile'], jsons_path = _path(opt)
        self._setup_data(opt['datafile'], jsons_path)
        self.id = 'ftml'
        self.reset()
        self.restricted_to_domain = None
        
        self.added_domains_buffer = {}
        
        

    def _setup_data(self, data_path, jsons_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.messages = json.load(data_file)

        test_path = os.path.join(jsons_path, 'testListFile.json')
        valid_path = os.path.join(jsons_path, 'valListFile.json')
        if self.datatype.startswith('test'):
            with open(test_path) as f:
                test_data = {line.strip(): self.messages[line.strip()] for line in f}
                self.messages = test_data
        elif self.datatype.startswith('valid'):
            with open(valid_path) as f:
                valid_data = {line.strip(): self.messages[line.strip()] for line in f}
                self.messages = valid_data
        else:
            with open(test_path) as f:
                for line in f:
                    if line.strip() in self.messages:
                        del self.messages[line.strip()]
            with open(valid_path) as f:
                for line in f:
                    if line.strip() in self.messages:
                        del self.messages[line.strip()]
        self.messages = list(self.messages.values())
        
        
        self.domains = [x for x in self.messages[0]['goal'] if self.messages[0]['goal'][x] == {} or 'info' in self.messages[0]['goal'][x]]
        domains = [d for i in range(len(self.messages)) for d in self.domains if len(self.messages[i]['goal'][d]) > 0]
        
        self.domain_convo_inds = {}
        for d in self.domains:
            self.domain_convo_inds[d] = [i for i in range(len(self.messages)) if domains[i] == d]
    

    def fix_teacher_domain(self, domains):
        self.restricted_to_domain = domains
    
    
    def add_domain(self, domain):
        self.added_domains_buffer[domain] = 0
        # shuffle the order the data will be accumulated in.
        random.shuffle(self.domain_convo_inds[domain])
    
        
    def added_domains(self):
        return [domain for domain in self.added_domains_buffer.keys() if self.added_domains_buffer[domain] > 0]
        
        
    def add_all_domain_data(self, domain):
        self.add_training_data(domain, len(self.domain_convo_inds[domain]))
        
                            
                            
    def add_training_data(self, domain, n):
        tot = len(self.domain_convo_inds[domain])
        accum = n + self.added_domains_buffer[domain]
#         print(tot, n, domain, self.added_domains_buffer[domain])
#         sys.exit()
        self.added_domains_buffer[domain] = min(tot, accum)
        
    def set_parley_inds(self, eps_idxs):
        self.parley_eps_idxs = eps_idxs        



    def num_examples(self):
        # This is in the whole dataset split, across domains.
        examples = 0
        for data in self.messages:
            examples += len(data['log']) // 2
        return examples



    def num_episodes(self):
        return len(self.messages)
            
            
            
    def num_episodes_in_restricted_domain(self):
        if self.restricted_to_domain: 
            return sum([len(self.domain_convo_inds[k]) for k in self.restricted_to_domain])
        else:
            return None
    
    
    
#     def set_parley_episode_idxs(idxs):
#         self.parley_episode_idxs = idxs
#         self.has_more_exs = True
    
    
    
    # From multiwoz teacher
    def get(self, episode_idx, entry_idx=0):
        log_idx = entry_idx * 2
        entry = self.messages[episode_idx]['log'][log_idx]['text']
        episode_done = log_idx == len(self.messages[episode_idx]['log']) - 2
        action = {
            'id': self.id,
            'text': entry,
            'episode_done': episode_done,
            'labels': [self.messages[episode_idx]['log'][log_idx + 1]['text']],
        }
        return action
        
        
    def get_episode(self, i):
        episode_idx = i
        actions = []
        # log_idx = entry_idx * 2
        for log_idx in range(0, len(self.messages[episode_idx]['log']), 2):
            entry = self.messages[episode_idx]['log'][log_idx]['text']
            episode_done = log_idx == len(self.messages[episode_idx]['log']) - 2
            action = {
                'id': self.id,
                'text': entry,
                'episode_done': episode_done,
                'labels': [self.messages[episode_idx]['log'][log_idx + 1]['text']],
            }
            actions.extend([action,])
        return actions
    
    def is_restricted(self):
        return self.restricted_to_domain is not None
    
        
        

### FILL IN THE BELOW METHODS       
#     def act(self):
#         """
#         Send new dialog message.
#         """
#         if not hasattr(self, 'epochDone'):
#             # reset if haven't yet
#             self.reset()
# 
#         # get next example, action is episode_done dict if already out of exs
#         action, self.epochDone = self.next_example()
#         # TODO: all teachers should eventually create messages
#         # while setting up the data, so this won't be necessary
#         action = Message(action)
#         action.force_set('id', self.getID())
# 
#         # remember correct answer if available
#         self.last_act = action
#         self.lastY = action.get('labels', action.get('eval_labels', None))
#         if not DatatypeHelper.is_training(self.datatype) and 'labels' in action:
#             # move labels to eval field so not used for training
#             # but this way the model can use the labels for perplexity or loss
#             action = action.copy()
#             labels = action.pop('labels')
#             if not self.opt.get('hide_labels', False):
#                 action['eval_labels'] = labels
# 
#         return action
        
    def get_added_episodes(self):
    
        added_episodes = []
        for k in self.restricted_to_domain:
            added_episodes += self.domain_convo_inds[k][:self.added_domains_buffer[k]]
            
        return added_episodes
                
    def next_episode_idx(self, num_eps=None, loop=None):
        """
        Return the next episode index.
        :param num_eps:
            default None uses ``num_episodes`` value.
        :param loop:
            default None loops during training but not evaluation.
        """
        if num_eps is None:
            num_eps = self.num_episodes()
        if loop is None:
            loop = self.training
            
        if self.is_restricted():
            # can be list of domains.
            added_episodes_domain = self.get_added_episodes()
            num_eps_domain = len(added_episodes_domain)
            
#         else:
#             # teacher is not restricted to a domain
#             print('teacher not restricted to a domain. Index: ', self.index.value)
            
                        
        if self.random:
#             print('random')
            # todo: should these be sampled without replacement?! Kun
            if self.is_restricted():
                new_idx = random.sample(added_episodes_domain, 1)[0]
            else:
                new_idx = random.randrange(num_eps)
        else:
#             print('streaming, loop:', loop)
            with self._lock():
                self.index.value += 1
                if loop:
                    # during training
                    if self.is_restricted():
                        self.index.value %= num_eps_domain
                    else: 
                        self.index.value %= num_eps
                    new_idx = self.index.value
                if self.is_restricted():
                    if self.index.value >= num_eps_domain:
                        new_idx = None
                    else:
                        new_idx = added_episodes_domain[self.index.value]
                else:
                    if self.index.value >= self.num_episodes():
                        new_idx = None
                    else:
                        new_idx = self.index.value
#         print('New_idx: ', new_idx, self.index.value)        
        return new_idx
        
        
        
    def has_more_exs(self):
        return self.has_more_exs
        
        
        
    def next_example(self):
        """
        Return the next example.
        If there are multiple examples in the same episode, returns the next one in that
        episode. If that episode is over, gets a new episode index and returns the first
        example of that episode.
        """
        
        if self._episode_done:
            self.episode_idx = self.next_episode_idx()
            self.entry_idx = 0
        else:
            self.entry_idx += 1
#         print('next_example info: ', self._episode_done, self.episode_idx, self.entry_idx)
        # HERE: This is what's returning empty examples.... Can I just remove this? How to replace?
        # Note: self.episode_idx indexes all the episodes. If teacher is constrained to a domain,
        # then self.num_episodes is episodes in the domain, and episode_idx can be larger.
        # However, self.index.value shouldn't be larger than the number of episodes in the domain...
        # This may need to change if we want to loop over the domain examples multiple times, e.g., fine-tuning.
        if self.episode_idx is None or self.episode_idx >= self.num_episodes():
            self.has_more_exs = False
            return {'episode_done': True}, True
#         print(self.index.value, self.num_episodes(), self.num_episodes_in_restricted_domain())

        if self.is_restricted() and self.index.value >= self.num_episodes_in_restricted_domain():
            print('no more episodes')
            self.has_more_exs = False
            return {'episode_done': True}, True

        ex = self.get(self.episode_idx, self.entry_idx)
#         k = self.restricted_to_domain
#         print('restricted episodes: ', self.domain_convo_inds[k][:self.added_domains_buffer[k]])
#         print('getting episode: %s, entry: %s' % (self.episode_idx, self.entry_idx))
#         print('TEACHER EXAMPLE: ', ex)
        self._episode_done = ex.get('episode_done', False)

        if (
            not self.cycle
            and self._episode_done
            and self.episode_idx + self.opt.get("batchsize", 1) >= self.num_episodes()
        ):
            epoch_done = True
        else:
            epoch_done = False
        
#         if self.episode_idx == self.parley_episode_idxs[-1] and epoch_done:
        return ex, epoch_done        


class DefaultTeacher(FtmlTeacher):
    def __init__(self, opt, shared=None):
        print('FTML TEACHER!')
        super().__init__(opt, shared=None)
    pass