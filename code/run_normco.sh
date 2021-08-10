DATA_DIR='../data/datasets'

MAX_DEPTH=5
MAX_NODES=50
SEARCH_METHOD=bfs
MODEL=LSTM
NUM_EPOCHS=100
BATCH_SIZE=32
SEQUENCE_LEN=5
EMBEDDING_DIM=128
LR=0.001
EVAL_EVERY=20

python run_normco.py\
  --data_dir $DATA_DIR\
	--max_depth $MAX_DEPTH\
	--max_nodes $MAX_NODES\
	--search_method $SEARCH_METHOD\
	--model $MODEL\
	--num_epochs $NUM_EPOCHS\
	--batch_size $BATCH_SIZE\
	--sequence_len $SEQUENCE_LEN\
	--lr $LR\
	--eval_every $EVAL_EVERY
