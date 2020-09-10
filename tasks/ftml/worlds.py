# from copy import deepcopy
# import json
# import random
# import os
# import string
# 
import sys
import copy
# from parlai.core.agents import create_agent
# from parlai.core.message import Message
from parlai.core.worlds import DialogPartnerWorld #, validate
# from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
# from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
# from parlai.utils.misc import warn_once
import random
from parlai.core.message import Message
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

            # Kun: these should be sampled without replacement, right? 
            # Kun: currently they sample D_tr from the training set and D_val from the validation set. Should we pool train & val?
            Dk_tr = random.sample(added_episodes_domain, self.opt.get('meta_batchsize_tr'))
            Dk_val = random.sample(added_episodes_domain, self.opt.get('meta_batchsize_val'))
            
            #set the parley data and then parley through them.
            self.update_parley(k, Dk_tr, Dk_val)
                
                
            
    def update_parley(self, k, Dk_tr, Dk_val):
        
        teacher, student = self.agents
        
        for i in range(len(Dk_tr)): 
            
            observations_tr = []; observations_val = []
            for a in teacher.get_episode(Dk_tr[i]):
                observations_tr.extend([student.observe(a)])
                student.self_observe(observations_tr[-1])
                
            for a in teacher.get_episode(Dk_val[i]):
                observations_val.extend([student.observe(a)])
                student.self_observe(observations_val[-1])
            
            
            batch_reply = [
                Message({'id': self.getID(), 'episode_done': False}) for _ in observations_tr
            ]
#             batch_reply_val = [
#                 Message({'id': self.getID(), 'episode_done': False}) for _ in observations_val
#             ]

            # check if there are any labels available, if so we will train on them
            self.is_training = True #any('labels' in obs for obs in observations)

            # create a batch from the vectors
            batch = student.batchify(observations_tr)
            batch_val = student.batchify(observations_val)
            
            print('EXAMPLES OF TRAINING OBSERVATIONS: ', len(observations_tr), observations_tr[:3])
            print('EXAMPLES OF VALIDATION OBSERVATIONS: ', len(observations_val), observations_val[:3])

#             if (
#                 'label_vec' in batch
#                 and 'text_vec' in batch
#                 and batch.label_vec is not None
#                 and batch.text_vec is not None
#             ):
#                 # tokens per batch
#                 # we divide by the binary is_primary_worker() so that the numerator is
#                 # num_tokens in all workers, and the denominator is 1.
#                 tpb = GlobalAverageMetric(
#                     (batch.label_vec != self.NULL_IDX).sum().item(),
#                     float(is_primary_worker()),
#                 )
#                 self.global_metrics.add('tpb', tpb)
# 
#             if self.is_training:
#             output = self.train_step(batch)
            
            student._init_cuda_buffer(self.opt['batchsize'], student.label_truncate or 256)
            student.model.train()
            student.zero_grad()
            
            init_model_params = copy.deepcopy(student.model.state_dict())
            param_names = list(init_model_params.keys())
            
            import torch; torch.set_printoptions(precision=6)
            print('Init: ', student.model.state_dict()[param_names[3]][0,:10])
            
            student._control_local_metrics(disabled=True) # turn off local metric computation
            loss = student.compute_loss(batch)
            
            student.backward(loss)
            student.update_params()
#             print('grad: ', list(student.model.parameters())[0].grad)
            
            print('Updated: ', student.model.state_dict()[param_names[3]][0,:10])
            
            loss = student.compute_loss(batch_val)
            student.backward(loss)
            print('second eval:', student.model.state_dict()[param_names[3]][0,:10])
            
            
            student.model.load_state_dict(init_model_params)
            print('back to init: ', student.model.state_dict()[param_names[3]][0,:10])
            
            
            student.update_params()
            print('final updated: ', student.model.state_dict()[param_names[3]][0,:10])
            
              
#             else:
#                 with torch.no_grad():
#                     # save memory and compute by disabling autograd.
#                     # use `with torch.enable_grad()` to gain back gradients.
#                     output = self.eval_step(batch)
# 
#             if output is not None:
#                 # local metrics are automatically matched up
#                 self.match_batch(batch_reply, batch.valid_indices, output)
# 
#             # broadcast the metrics back
#             for k, values in self._local_metrics.items():
#                 if len(values) != len(batch.valid_indices):
#                     raise IndexError(
#                         f"Batchsize mismatch on metric {k} (got {len(values)}, "
#                         f"expected {len(batch.valid_indices)}"
#                     )
#                 for i, value in zip(batch.valid_indices, values):
#                     if 'metrics' not in batch_reply[i]:
#                         batch_reply[i]['metrics'] = {}
#                     batch_reply[i]['metrics'][k] = value
# 
#             # Make sure we push all the metrics to main thread in hogwild/workers
#             self.global_metrics.flush()

        return batch_reply        
        
        

        
        
#         for a in teacher_acts_tr:
#             agents[1].observe(validate(a))
#             discard_response = agents[1].act() # backward inside train_step()
#             
#         for a in teacher_acts_val:
#             agents[1].observe(validate(a))
#             discard_response = agents[1].act()
        
        # agents[1].observe(validate(acts[0]))
#         agents[1].observe_tr_val(teacher_acts_tr, teacher_acts_val)
        
        # acts[1] = agents[1].act()
#         student_acts = 
#         agents[1].meta_act()
        
        # todo, does the teacher need to observe?
#         agents[0].observe(validate(acts[1]))
#         self.update_counters()




