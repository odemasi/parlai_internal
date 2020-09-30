from parlai.scripts.train_model import TrainLoop, TrainModel, load_eval_worlds, setup_args as setup_train_args
import sys

# import json
# import numpy as np
# import os
# import signal
# from typing import Dict
# 
# from parlai.core.metrics import Metric
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.exceptions import StopTrainException
from parlai.core.logs import TensorboardLogger
# from parlai.core.metrics import aggregate_named_reports, aggregate_unnamed_reports
from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.worlds import create_task
# from parlai.scripts.build_dict import build_dict, setup_args as setup_dict_args
from parlai.utils.distributed import (
    sync_object,
    is_primary_worker,
    all_gather_list,
    is_distributed,
    num_workers,
)
# from parlai.utils.misc import Timer, nice_report
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import copy
import pickle
import random



def setup_args(parser=None) -> ParlaiParser:
    """
    Build the ParlAI parser, adding command line args if necessary.
    :param ParlaiParser parser:
        Preexisting parser to append options to. Will be created if needed.
    :returns:
        the ParlaiParser with CLI options added.
    """
    if parser is None:
        parser = ParlaiParser(True, True, 'Train a model')
    train = parser.add_argument_group('FTML Training Loop Arguments')

#     train.add_argument('--n_grad', type='bool', default=False, hidden=True)
    train.add_argument('-nomt', '--no-multi-task', type='bool', default=False)
    
    TensorboardLogger.add_cmdline_args(parser)

    parser = setup_train_args(parser)
    return parser
    
    
    
class MtftTrainLoop(TrainLoop):
    """
    MtftTrainLoop contains the core training loop logic.
    """

    def __init__(self, opt):
        super().__init__(opt)
        # Create model and assign it to the specified task
#         self.agent = create_agent(opt)
#         self.agent.opt.log()
#         self.world = create_task(opt, self.agent)
        
#         print('should be DefaultWorld: ', self.world.__class__.__name__)
#         print('should be DefaultTeacher: ', self.world.agents[0].__class__.__name__)
#         print('should be FtmlLearnerAgent: ', self.world.agents[1].__class__.__name__)
        
        
        self.test_worlds = load_eval_worlds(self.agent, opt, 'test')
        self.valid_worlds = load_eval_worlds(self.agent, opt, 'valid')
        self.logging_filename = opt['logging_filename']
        
        # smart defaults for --validation-metric-mode
#         if opt['validation_metric'] in {'loss', 'ppl', 'mean_rank'}:
#             opt['validation_metric_mode'] = 'min'
#         elif opt['validation_metric'] in {'accuracy', 'hits@1', 'hits@5', 'f1', 'bleu'}:
#             opt['validation_metric_mode'] = 'max'
#         if opt.get('validation_metric_mode') is None:
#             opt['validation_metric_mode'] = 'max'

    def write_log(self, log_text):
        with open(self.logging_filename, 'a') as f:
            f.write(log_text+'\n')

    def train(self):
        """
        Perform ftml training run.
        :return: tuple of reports (validation_report, test_report)
        """
        logging.info('training...')
        opt = self.opt
        world = self.world
        teacher = world.agents[0]
        more_data_in_domain = True
        
        eval_data = {x:[] for x in teacher.domains}
        
        with world:
            shuffled_domains = [x for x in teacher.domains if x not in ['police', 'hospital']]
            random.shuffle(shuffled_domains)
            
            for d, domain in enumerate(shuffled_domains):
                

                N = len(teacher.domain_convo_inds[domain])
                teacher.add_domain(domain)
                teacher.add_all_domain_data(domain)
                
            self.best_valid = None
            stop_training = False
            
            if opt['no_multi_task']:
                # only fine-tune to each domain, so don't enter the multi-task while loop
                stop_training = True
                self._total_epochs = 0
                self._total_exs = 0
                
            while not stop_training: # This is for multi-tasking a global model
                for _ in range(int(teacher.num_episodes())): 
                    world.parley()
                    self.parleys += 1
                    
                    # TODO: I think this should be set correctly. Helps tracking, needed for termination?
                    self._total_epochs = 0
                    self._total_exs = 0
                    
                    train_time, log_time, validate_time = sync_object(
                        (
                            self.train_time.time(),
                            self.log_time.time(),
                            self.validate_time.time(),
                        )
                    )
                    
                    if log_time > self.log_every_n_secs:
                        self.log()
                self.write_log('finished %s parleys' % self.parleys)
                self.write_log('Learning rate before valid: %s ' % world.agents[1].optimizer.state_dict()['param_groups'][0]['lr'])
                # validation_decreasing = # todo
                # todo: add validation here to tell when to stop updating the meta model.
                # This is harder to do, as the validation is of the meta model....
                for w in self.valid_worlds:
                    for dd in teacher.added_domains(): 
                        w.reset() # Should also reset the teacher.index.value --> -1, but keep the domain fixed.
                        w.agents[0].add_domain(dd)
                        w.agents[0].add_all_domain_data(dd)
                    # Fix validation teacher to domains training teacher has seen.
                    w.agents[0].fix_teacher_domain(teacher.added_domains()) 
                    w.agents[0].index.value = -1 # reset index because we'll stream through the training data.
                    w.agents[0].entry_idx = 0
                    
                stop_training = self.validate()
                logging.info('Multi-task model validation value: %s ' % self.best_valid)

                    
                
            # After the multi-task model is trained, fine tune model for each domain.
            M = copy.deepcopy(world.agents[1].model.state_dict())
            for dd in teacher.domains: 
                
                # Restrict valid_world teachers to chosen domain for fine-tuning
                for w in self.valid_worlds:
                    w.reset() # Should also reset the teacher.index.value --> -1, but keep the domain fixed.
                    w.agents[0].add_domain(dd)
                    w.agents[0].add_all_domain_data(dd)
                    w.agents[0].fix_teacher_domain([dd]) 
                    w.agents[0].index.value = -1 # reset index because we'll stream through the data.
                    w.agents[0].entry_idx = 0
                
                # Restrict test worlds to chosen domain for fine-tuning
                for w in self.test_worlds:
                    w.reset() # Should also reset the teacher.index.value --> -1, but keep the domain fixed.
                    w.agents[0].add_domain(dd)
                    w.agents[0].add_all_domain_data(dd)
                    w.agents[0].fix_teacher_domain([dd])
                    w.agents[0].index.value = -1 # reset index because we'll stream through the data.
                    w.agents[0].entry_idx = 0
                
                    # note the appropriate state_dict should be loaded, as the agent should 
                    # be shared by reference in the training and the testing worlds.
                    
                self.write_log("STARTING Fine-tuning OF STUDENT HERE on %s " % dd)
                if self.test_worlds[0].agents[0].num_episodes_in_restricted_domain() > 0:                
                    # make sure the meta parameters are loaded before evaluating another training domain
                    world.agents[1].model.load_state_dict(M)
                    
                    # Restrict training world to fine-tuning domain
                    teacher.fix_teacher_domain([dd])
                    teacher.index.value = -1 # reset index because we'll stream through the training data.
                    teacher.entry_idx = 0   
                    
                    
                    logging.info('Fine-tuning to: %s'% dd)
                    self.best_valid = None
                    stop_training = False
                    self.tune_parley_epochs = 0
                    
                    # Fine-tune model to single domain
                    while not stop_training:
                        # fine-tune for one epoch over training
                        self.write_log('Learning rate : %s ' % world.agents[1].optimizer.state_dict()['param_groups'][0]['lr'])
                    
    #                     while not world.epoch_done(): # HERE: loop for an epoch over domain training data. 
                        for n in range(teacher.num_episodes_in_restricted_domain()): # epoch episodes, as each full episode processed. 
                            world.parley() # Note the updating is fixed to the domain training data only.
                        
                        # fine-tune until validation on domain stops decreasing.
                        stop_training = self.validate()
                        self.tune_parley_epochs += 1
                        
                        logging.info('Best valid: %s' % self.best_valid)
                        self.write_log('Best fine-tune valid: %s' % self.best_valid)
                        self.write_log('Finished %s tune_parleys' % self.tune_parley_epochs)
                        
                        
                    # Evaluate on domain test set.
                    max_exs = -1
                    t_report = self._run_eval(self.test_worlds, opt, 'test', max_exs, write_log=True)
                    logging.info('on domain %s: test report: ' % dd)
                    logging.info(t_report) 
                    eval_data[dd] = {'domain':dd, 'report': t_report, 'num_parleys': self.parleys, 'tune_epochs': self.tune_parley_epochs}
                
        
        import datetime
        stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        if opt['no_multi_task']:
            locationname = '/home/oademasi/transfer-learning-conv-ai/ParlAI/parlai_internal/eval_data_ft_%s.pkl' % stamp
        else:
            locationname = '/home/oademasi/transfer-learning-conv-ai/ParlAI/parlai_internal/eval_data_mtft_%s.pkl' % stamp
        pickle.dump(eval_data, open(locationname, 'wb'))
        print('wrote to: ', locationname)
        v_report = None
        t_report = None
        return v_report, t_report


@register_script('mtft_train_model', aliases=['mtfttm', 'mtft_train'])
class MtftTrainModel(TrainModel):
    
    @classmethod
    def setup_args(cls):
        return setup_args()
        
    def run(self):
        import os
        model_file = self.opt['model_file']
        for f in (model_file, '%s.opt' % model_file, '%s.trainstats' % model_file):
            fdir = '/home/oademasi/transfer-learning-conv-ai/ParlAI/'
            fname = fdir + f
            if os.path.isfile(fname):
                os.remove(fname)
                
        import datetime
        stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        if self.opt['no_multi_task']:
            self.opt['logging_filename'] = 'logging_ft_%s.txt' % stamp
        else:
            self.opt['logging_filename'] = 'logging_mtft_%s.txt' % stamp
        self.train_loop = MtftTrainLoop(self.opt)
        return self.train_loop.train()


if __name__ == '__main__':
    MtftTrainModel.main()
#     python parlai_internal/scripts/train_ftml.py -t internal:ftml --model internal:ftml_learner --model_file discard
    # python parlai_internal/scripts/train_ftml.py -t internal:ftml --model image_seq2seq --model_file discard