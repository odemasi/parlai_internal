
# from parlai.agents.image_seq2seq.image_seq2seq import ImageSeq2seqAgent
# from .build import build
import os
import json
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
# from parlai.agents.image_seq2seq.image_seq2seq import ImageSeq2seqAgent
from parlai.core.message import Message
import random
import sys



class FtmlLearnerAgent(Seq2seqAgent):
    
#     def __init__(self, opt):
#         super().__init__(opt)
        
        
    def observe_tr_val(self, observations_tr, observations_val):
        """
        Process incoming message in preparation for producing a response.
        This includes remembering the past history of the conversation.
        """
        # TODO: Migration plan: TorchAgent currently supports being passed
        # observations as vanilla dicts for legacy interop; eventually we
        # want to remove this behavior and demand that teachers return Messages
        observations_tr = [Message(observation) for observation in observations_tr]
        observations_val = [Message(observation) for observation in observations_val]

        # Sanity check everything is in order
        self._validate_observe_invariants()

        print('Need to check if episodes done in FtmlLearnerAgent!')
#         if observation.get('episode_done'):
#             self.__expecting_clear_history = True
#         elif 'labels' in observation or 'eval_labels' in observation:
#             # make sure we note that we're expecting a reply in the future
#             self.__expecting_to_reply = True

        self.observations_tr = observations_tr
        self.observations_val = observations_val
        print('Observations in tr set:')
        print(self.observations_tr)
        
        print('\n\nObservations in val set:')
        print(self.observations_val)
        sys.exit()
        # Update the history using the observation.
        # We may also consider adding a temporary string to the history
        # using the `get_temp_history()` function: this string will
        # persist until it is updated.
#         self.history.update_history(
#             observation, temp_history=self.get_temp_history(observation)
#         )
#         return self.vectorize(
#             observation,
#             self.history,
#             text_truncate=self.text_truncate,
#             label_truncate=self.label_truncate,
#         )
        
        
    def meta_act(self):
    # def act(self):
        """
        Call batch_act with the singleton batch.
        """
        # BatchWorld handles calling self_observe, but we're in a Hogwild or Interactive
        # world, so we need to handle this ourselves.
#         response = 
        self.meta_batch_act(self.observations_tr, self.observation_val)#[0]
        
#         self.self_observe(response)
#         return response
        
        
        
        
    def meta_batch_act(self, observations_tr, observations_val):
        """
        Process a batch of observations (batchsize list of message dicts).
        These observations have been preprocessed by the observe method.
        Subclasses can override this for special functionality, but if the
        default behaviors are fine then just override the ``train_step`` and
        ``eval_step`` methods instead. The former is called when labels are
        present in the observations batch; otherwise, the latter is called.
        """
        # clear local metrics before anything else
        self._local_metrics.clear()

        # initialize a list of replies with this agent's id
        batch_reply = [
            Message({'id': self.getID(), 'episode_done': False}) for _ in observations_tr
        ]
        batch_reply_val = [
            Message({'id': self.getID(), 'episode_done': False}) for _ in observations_val
        ]

        # check if there are any labels available, if so we will train on them
        self.is_training = True #any('labels' in obs for obs in observations)

        # create a batch from the vectors
        batch = self.batchify(observations_tr)
        batch_val = self.batchify(observations_val)

        if (
            'label_vec' in batch
            and 'text_vec' in batch
            and batch.label_vec is not None
            and batch.text_vec is not None
        ):
            # tokens per batch
            # we divide by the binary is_primary_worker() so that the numerator is
            # num_tokens in all workers, and the denominator is 1.
            tpb = GlobalAverageMetric(
                (batch.label_vec != self.NULL_IDX).sum().item(),
                float(is_primary_worker()),
            )
            self.global_metrics.add('tpb', tpb)

        if self.is_training:
            output = self.meta_train_step(batch, batch_val)
        else:
            print('Why is meta_batch_act in eval mode?')
            sys.exit()
            
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back gradients.
                output = self.eval_step(batch)

        if output is not None:
            # local metrics are automatically matched up
            self.match_batch(batch_reply, batch.valid_indices, output)

        # broadcast the metrics back
        for k, values in self._local_metrics.items():
            if len(values) != len(batch.valid_indices):
                raise IndexError(
                    f"Batchsize mismatch on metric {k} (got {len(values)}, "
                    f"expected {len(batch.valid_indices)}"
                )
            for i, value in zip(batch.valid_indices, values):
                if 'metrics' not in batch_reply[i]:
                    batch_reply[i]['metrics'] = {}
                batch_reply[i]['metrics'][k] = value

        # Make sure we push all the metrics to main thread in hogwild/workers
        self.global_metrics.flush()

        return batch_reply
        
        
    def meta_train_step(self, batch, batch_val):
        """
        Train on a single batch of examples.
        """
        # helps with memory usage
        # note we want to use the opt's batchsize instead of the observed batch size
        # in case dynamic batching is in use
        self._init_cuda_buffer(self.opt['batchsize'], self.label_truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            loss = self.compute_loss(batch)
            self.backward(loss)
            
            loss_val = self.compute_loss(batch_val)
            self.backward(loss_val) 
            
            # Kun: todo: call update_params between steps?
            self.update_params()
            oom_sync = False
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                oom_sync = True
                logging.error(
                    'Ran out of memory, skipping batch. '
                    'if this happens frequently, decrease batchsize or '
                    'truncate the inputs to the model.'
                )
                self.global_metrics.add('skipped_batches', SumMetric(1))
            else:
                raise e

        if oom_sync:
            # moved outside of the try-except because the raised exception in scope
            # actually prevents from the data being freed, which can sometimes cause
            # us to OOM during our OOM handling.
            # https://github.com/pytorch/pytorch/issues/18853#issuecomment-583779161

            # gradients are synced on backward, now this model is going to be
            # out of sync! catch up with the other workers
            self._init_cuda_buffer(8, 8, True)
        
        
        
    



def _path(opt):
    # build the data if it does not exist
#     build(opt)

    # set up path to data (specific to each dataset)
    print('DATAPATH:', opt['datapath'])
    jsons_path = os.path.join(opt['datapath'], 'multiwoz', 'MULTIWOZ2.1')
    conversations_path = os.path.join(jsons_path, 'data.json')
    return conversations_path, jsons_path

