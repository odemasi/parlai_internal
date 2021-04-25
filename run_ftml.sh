

VALIDATION_MAX_EX=100
DEBUG_FLAGS='--validation-max-exs '$VALIDATION_MAX_EX

META_BATCH_SIZE=32
NUM_META_STEPS=100

# FTML: incrementally meta-update meta model + fine tune to domain
# CUDA 4 --> GPU 3
# CUDA 7 --> GPU 7
# CUDA 1 --> GPU 5
# CUDA 2 --> GPU 0
CUDA_VISIBLE_DEVICES=1 python parlai_internal/scripts/train_ftml.py -t internal:ftml --model internal:ftml_learner --model_file discard_ftml --metrics ppl,f1,accuracy,bleu --validation-metric loss --validation-patience 2 --validation_every_n_epochs 1 -lr 0.5 --load-from-checkpoint False --log-every-n-secs 120 --num-meta-steps $NUM_META_STEPS --meta-batchsize_tr $META_BATCH_SIZE --meta-batchsize_val $META_BATCH_SIZE --num-episode-batch 10 --eval-batch-size 10 -lr 7e-2 --lr-scheduler none --optimizer adamax $DEBUG_FLAGS 

# 100 meta steps / batch size 10 episodes should give an epoch of about 1000 examples.


