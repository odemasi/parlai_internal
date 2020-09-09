# from copy import deepcopy
# import json
# import random
# import os
# import string
# 
# 
# from parlai.core.agents import create_agent
# from parlai.core.message import Message
from parlai.core.worlds import DialogPartnerWorld #, validate
# from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
# from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
# from parlai.utils.misc import warn_once
import random
from parlai.core.worlds import validate



class DefaultWorld(DialogPartnerWorld):
    """
    Interactive world for FTML training
    Specifically a world for models trained on the task `-t ftml`.
    """

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print('FTML WORLD!!!!')        
        
        
    def meta_parleys(self, n_meta):
        # meta_update
        
        teacher = self.agents[0]
        for nm in range(self.opt.get('n_meta_steps')):
            
            # Sample domain to take data from
            k = random.choice(list(teacher.added_domains_buffer.keys()))
            
            # Sample minibatches and set these as the parley data
            added_episodes_domain = teacher.domain_convo_inds[k][:teacher.added_domains_buffer[k]]

            # todo: should these be sampled without replacement?! Kun
            Dk_tr = random.sample(added_episodes_domain, self.opt.get('meta_batchsize_tr'))
            Dk_val = random.sample(added_episodes_domain, self.opt.get('meta_batchsize_val'))
            
            #set the parley data and then parley through them.
            self.update_parley(k, Dk_tr, Dk_val)
                
                
            
    def update_parley(self, k, Dk_tr, Dk_val):
#         acts = self.acts
        agents = self.agents
        
        # acts[0] = agents[0].act()
        teacher_acts_tr = [validate(a) for i in Dk_tr for a in agents[0].get_episode(i)]
        teacher_acts_val = [validate(a) for j in Dk_val for a in agents[0].get_episode(j)]
        
        
        # agents[1].observe(validate(acts[0]))
        agents[1].observe_tr_val(teacher_acts_tr, teacher_acts_val)
        
        # acts[1] = agents[1].act()
#         student_acts = 
        agents[1].meta_act()
        
        # todo, does the teacher need to observe?
#         agents[0].observe(validate(acts[1]))
        self.update_counters()




