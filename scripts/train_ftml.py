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
# from parlai.utils.distributed import (
#     sync_object,
#     is_primary_worker,
#     all_gather_list,
#     is_distributed,
#     num_workers,
# )
# from parlai.utils.misc import Timer, nice_report
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import copy
import pickle



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
#     train.add_argument('-ngrad', '--num-grad', type=int, default=2)
#     train.add_argument('-nadd', '--num-added-data', type=int, default=50)
    train.add_argument('-mbchsztr', '--meta-batchsize_tr', type=int, default=10)
    train.add_argument('-mbchszval', '--meta-batchsize_val', type=int, default=10)
    train.add_argument('-nmmetastep', '--num-meta-steps', type=int, default=50)
    
    TensorboardLogger.add_cmdline_args(parser)

    parser = setup_train_args(parser)
    return parser
    
    
    
class FtmlTrainLoop(TrainLoop):
    """
    FtmlTrainLoop contains the core training loop logic.
    """

    def __init__(self, opt):
        super().__init__(opt)
        # Create model and assign it to the specified task
#         self.agent = create_agent(opt)
#         self.agent.opt.log()
#         self.world = create_task(opt, self.agent)
        self.meta_parleys = 0
#         print('should be DefaultWorld: ', self.world.__class__.__name__)
#         print('should be DefaultTeacher: ', self.world.agents[0].__class__.__name__)
#         print('should be FtmlLearnerAgent: ', self.world.agents[1].__class__.__name__)
        
        
        self.test_worlds = load_eval_worlds(self.agent, opt, 'test')
        self.valid_worlds = load_eval_worlds(self.agent, opt, 'valid')
        
        
        # smart defaults for --validation-metric-mode
#         if opt['validation_metric'] in {'loss', 'ppl', 'mean_rank'}:
#             opt['validation_metric_mode'] = 'min'
#         elif opt['validation_metric'] in {'accuracy', 'hits@1', 'hits@5', 'f1', 'bleu'}:
#             opt['validation_metric_mode'] = 'max'
#         if opt.get('validation_metric_mode') is None:
#             opt['validation_metric_mode'] = 'max'


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
        
        eval_data = []
        
        with world:
            print(teacher.domains)
            for d, domain in enumerate(teacher.domains):
                
                eval_data.append({x:[] for x in teacher.domains[:d]})

                N = len(teacher.domain_convo_inds[domain])
                teacher.add_domain(domain)
                teacher.add_all_domain_data(domain)
#                 more_data_in_domain = True
                self.best_valid = None
                stop_training = False
                
#                 while more_data_in_domain:
                while not stop_training:
#                  while validation_decreasing: # Kun: this doesn't make sense without fine-tuning 
#                  the meta model first. We need another way to decide to stop meta-updates.
#                     teacher.add_training_data(domain, opt.get('num_added_data'))
                    
#                     logging.info('%s episodes of %s total training episodes added' % (teacher.added_domains_buffer[domain], N))
                    # do one batch of examples
                    # meta_update
                    world.meta_parleys()
                    
                    
                    # TODO: I think this should be set correctly. Helps tracking, needed for termination?
                    self._total_epochs = 0
                    self._total_exs = 0
                    # get the total training examples done, compute epochs
#                     self._total_epochs = self._preempted_epochs + sum(
#                         all_gather_list(world.get_total_epochs())
#                     )
#                     exs_per_epoch = world.num_examples()
#                     self._total_exs = int(np.round(self._total_epochs * exs_per_epoch))
#                     # and use the primary worker's timings for everything
#                     train_time, log_time, validate_time = sync_object(
#                         (
#                             self.train_time.time(),
#                             self.log_time.time(),
#                             self.validate_time.time(),
#                         )
#                     )
                
                
                    # validation_decreasing = # todo
                    # todo: add validation here to tell when to stop updating the meta model.
                    # This is harder to do, as the validation is of the meta model....
                    teacher.fix_teacher_domain(teacher.added_domains())
                    teacher.index.value = -1 # reset index because we'll stream through the training data.
                    teacher.entry_idx = 0
                    stop_training = self.validate()
                    logging.info('Meta-model validation value: %s ' % self.best_valid)
#                     more_data_in_domain = teacher.added_domains_buffer[domain] < N
                    
                    self.meta_parleys += 1
                    
                    
                # After the domain data has been accumulated, fine tune model for each domain.
                # i.e., update_procedure in Finn et al.
                # Kun: Finn et al. use this as a threshold to decide when to stop training. 
                # maybe we should only do when the domain data has been fully accumulated?
                
                M = copy.deepcopy(world.agents[1].model.state_dict())
                # fine-tune to each domain to test performance.
                logging.info('Added domains: %s ' % ', '.join(teacher.added_domains()))
#                 sys.exit()
                for dd in teacher.added_domains(): 
                    world.agents[1].model.load_state_dict(M)
                    logging.info('evaluating on domain %s...' % dd)
                    
                    teacher.fix_teacher_domain([dd])
                    teacher.index.value = -1 # reset index because we'll stream through the training data.
                    teacher.entry_idx = 0
                    
#                     logging.info("Num teacher episodes %s should be %s in test domain %s" % (teacher.num_episodes_in_restricted_domain(), len(teacher.domain_convo_inds[teacher.restricted_to_domain]), dd))
#                     for n in range(opt.get('num_grad')): # fine-tuning steps
                    logging.info('Fine-tuning to: %s'% dd)
                    self.best_valid = None
                    stop_training = False
                    self.tune_parley_epochs = 0
                    
                    while not stop_training:
                        # fine-tune for one epoch over training
                        for n in range(teacher.num_episodes_in_restricted_domain()): # epoch episodes, as each full episode processed. 
                            # Kun: what do you think about fixing domain-tuning to an epoch over 
                            # the train or train + val data?
                            world.parley() # Note the updating is fixed to the domain training data only.
                        # fine-tune until validation on domain stops decreasing.
                        stop_training = self.validate()
                        logging.info('Best valid: %s' % self.best_valid)
                        self.tune_parley_epochs += 1
                    
                    # if there is test data:
                    for w in self.test_worlds:
                        w.reset() # Should also reset the teacher.index.value --> -1, but keep the domain fixed.
                        w.agents[0].add_domain(dd)
                        w.agents[0].add_all_domain_data(dd)
                        w.agents[0].fix_teacher_domain([dd])
                    
                        # note the appropriate state_dict should be loaded, as the agent should 
                        # be shared by reference in the training and the testing worlds.
#                         logging.info("Num test episodes %s should be %s in domain %s" % (w.agents[0].num_episodes_in_restricted_domain(), len(w.agents[0].domain_convo_inds[w.agents[0].restricted_to_domain]), dd))
                    
                    print("STARTING EVALUATION OF STUDENT HERE")
                    if self.test_worlds[0].agents[0].num_episodes_in_restricted_domain() > 0:
                        max_exs = -1
                        t_report = self._run_eval(self.test_worlds, opt, 'test', max_exs, write_log=True)
                        logging.info('on domain %s: test report: ' % dd)
                        logging.info(t_report) 
                        eval_data[-1][dd] = {'domain': dd, 'report': t_report, 'meta_parleys': self.meta_parleys, 'tune_epochs': self.tune_parley_epochs}   
                        
                    # make sure the meta parameters are loaded before evaluating another training domain
                    world.agents[1].model.load_state_dict(M)

                # get the total training examples done, compute epochs
#                 self._total_epochs = self._preempted_epochs + sum(
#                     all_gather_list(world.get_total_epochs())
#                 )
#                 exs_per_epoch = world.num_examples()
#                 self._total_exs = int(np.round(self._total_epochs * exs_per_epoch))
                
#                 and use the primary worker's timings for everything
#                 train_time, log_time, validate_time = sync_object(
#                     (
#                         self.train_time.time(),
#                         self.log_time.time(),
#                         self.validate_time.time(),
#                     )
#                 )
# 
#                 check counters and timers
#                 if self._total_epochs >= self.max_num_epochs:
#                     self.log()
#                     logging.info(
#                         f'num_epochs completed:{self.max_num_epochs} time elapsed:{train_time}s'
#                     )
#                     break
#                 if train_time > self.max_train_time:
#                     logging.info(f'max_train_time elapsed:{train_time}s')
#                     break
#                 if log_time > self.log_every_n_secs:
#                     self.log()
#                 if (
#                     validate_time > self.val_every_n_secs
#                     or self._total_epochs - self.last_valid_epoch
#                     >= self.val_every_n_epochs
#                 ):
#                     try:
#                         log before we validate
#                         self.log()
#                         world.reset_metrics()
#                         stop_training = self.validate()
#                     except StopTrainException:
#                         if is_distributed():
#                             raise RuntimeError(
#                                 "StopTrainException not supported for distributed mode"
#                             )
#                         break
#                     reset the log time because we logged right before validating
#                     self.log_time.reset()
#                     self.last_valid_epoch = self._total_epochs
#                     if stop_training:
#                         break
#                     make sure metrics are clean before we log
#                     world.reset_metrics()
#                 if (
#                     self.save_time.time() > self.save_every_n_secs
#                     and opt.get('model_file')
#                     and is_primary_worker()
#                 ):
#                     logging.info(
#                         f"saving model checkpoint: {opt['model_file']}.checkpoint"
#                     )
#                     if opt['tensorboard_log'] and is_primary_worker():
#                         self.tb_logger.flush()
#                     self.save_model('.checkpoint')
#                     self.save_time.reset()

#         if not self.saved and is_primary_worker():
#             # save agent
#             self.save_model()

        # there's a rare edge case where the we never saved the model, and we try
        # # to reload it. This sync_object ensures all workers wait for the primary
        # worker to finish flushing before loading from disk.
#         sync_object(None)
#         if opt.get('model_file'):
#             # clean up all our memory, just to make sure we don't OOM on GPU when
#             # reloading the world
#             del world
#             del self.world
#             del self.agent
#             del self.valid_worlds
#             # reload best validation model
#             self.agent = create_agent(opt)

        # perform final validation/testing
#         valid_worlds = load_eval_worlds(self.agent, opt, 'valid')
#         max_exs = opt['validation_max_exs'] if opt.get('short_final_eval') else -1
#         v_report = self._run_eval(valid_worlds, opt, 'valid', max_exs, write_log=True)
#         test_worlds = load_eval_worlds(self.agent, opt, 'test')
#         t_report = self._run_eval(test_worlds, opt, 'test', max_exs, write_log=True)
#         if valid_worlds:
#             for valid_world in valid_worlds:
#                 valid_world.shutdown()
#         if test_worlds:
#             for test_world in test_worlds:
#                 test_world.shutdown()
# 
#         print_announcements(opt)
        
        import datetime
        stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        locationname = '/home/oademasi/transfer-learning-conv-ai/ParlAI/parlai_internal/eval_data_ftml_%s.pkl' % stamp
        pickle.dump(eval_data, open(locationname, 'wb'))
        v_report = None
        t_report = None
        return v_report, t_report


@register_script('ftml_train_model', aliases=['ftmltm', 'ftml_train'])
class FtmlTrainModel(TrainModel):
    
    @classmethod
    def setup_args(cls):
        return setup_args()
        
    def run(self):
        for i in range(10): 
            self.train_loop = FtmlTrainLoop(self.opt)
            self.train_loop.train()
        return 


if __name__ == '__main__':
    FtmlTrainModel.main()
#     python parlai_internal/scripts/train_ftml.py -t internal:ftml --model internal:ftml_learner --model_file discard
    # python parlai_internal/scripts/train_ftml.py -t internal:ftml --model image_seq2seq --model_file discard