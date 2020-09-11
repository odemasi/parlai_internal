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
    train.add_argument('-ngrad', '--num-grad', type=int, default=2)
    train.add_argument('-nadd', '--num-added-data', type=int, default=200)
    train.add_argument('-mbchsztr', '--meta-batchsize_tr', type=int, default=2)
    train.add_argument('-mbchszval', '--meta-batchsize_val', type=int, default=2)
    train.add_argument('-nmmetastep', '--num-meta-steps', type=int, default=2)
    
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
        print('should be DefaultWorld: ', self.world.__class__.__name__)
        print('should be DefaultTeacher: ', self.world.agents[0].__class__.__name__)
        print('should be FtmlLearnerAgent: ', self.world.agents[1].__class__.__name__)
        
        
        self.test_worlds = load_eval_worlds(self.agent, opt, 'test')
        
        
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
        domains = []; print('todo')
        teacher = world.agents[0]
        more_data_in_domain = True
        
        
        with world:
            for domain in teacher.domains:
            
                N = len(teacher.domain_convo_inds[domain])
                teacher.add_domain(domain)
                
                while more_data_in_domain:
                    teacher.add_training_data(domain, opt.get('num_added_data'))
                    
                    print('%s data of %s total episodes added' % (teacher.added_domains_buffer[domain], N))
#                     print("Num teacher episodes: ", teacher.num_episodes())
                    # do one example / batch of examples
                    # meta_update
                    world.meta_parleys()
#                     try:
#                         # meta_update
#                         world.meta_parleys(opt.get('num_meta_steps'))
#                     except StopTrainException:
#                         if is_distributed():
#                             raise RuntimeError(
#                                 "StopTrainException not supported for " "distributed mode"
#                             )
#                         break

                    self.meta_parleys += 1
                    
                    
                    # fine tune model.
                    # i.e., update_procedure
                    # Kun: Finn et al. use this as a threshold to decide when to stop training. 
                    # maybe we should only do when the domain data has been fully accumulated?
                    M = copy.deepcopy(world.agents[1].model.state_dict())
                    print("Num teacher episodes: ", teacher.num_episodes())
                    teacher.fix_teacher_domain(domain)
                    
                    print("Num teacher episodes %s should be %s in domain %s" % (teacher.num_episodes(), len(teacher.domain_convo_inds[teacher.restricted_to_domain]), domain))
                    for n in range(opt.get('num_grad')): 
                        # Kun: what do you think about fixing domain-tuning to an epoch over 
                        # the train or train + val data?
                        world.parley() # Note the updating is fixed to the domain training data only.
                    
                    
                    # if loss < gamma, record efficiency for domain as |Dt| datapoints
                    more_data_in_domain = teacher.added_domains_buffer[domain] < N
                    if not more_data_in_domain:
                        logging.info('evaluating on domain %s...' % domain)
                        print('todo: Record final performance of w˜_t on test set for task t.')
                        
                        for w in self.test_worlds:
                            w.reset() # Should also reset the teacher.index.value --> -1, but keep the domain set.
                            w.agents[0].add_domain_data(domain)
                            w.agents[0].fix_teacher_domain(domain)
                            print("Num test-teacher episodes %s should be %s in domain %s" % (w.agents[0].num_episodes(), len(w.agents[0].domain_convo_inds[w.agents[0].restricted_to_domain]), domain))
                        max_exs = -1
                        t_report = self._run_eval(self.test_worlds, opt, 'test', max_exs, write_log=True)
                        print(t_report)     
        
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

        v_report = None
        t_report = None
        return v_report, t_report


@register_script('ftml_train_model', aliases=['ftmltm', 'ftml_train'])
class FtmlTrainModel(TrainModel):
    
    @classmethod
    def setup_args(cls):
        return setup_args()
        
    def run(self):
        self.train_loop = FtmlTrainLoop(self.opt)
        return self.train_loop.train()


if __name__ == '__main__':
    FtmlTrainModel.main()
#     python parlai_internal/scripts/train_ftml.py -t internal:ftml --model internal:ftml_learner --model_file discard
    # python parlai_internal/scripts/train_ftml.py -t internal:ftml --model image_seq2seq --model_file discard