python prediction_train.py \
  --data_dir=push/push_train \ # path to the training set.
  --model=CDNA \ # the model type to use - DNA, CDNA, or STP
  --output_dir=./checkpoints \ # where to save model checkpoints
  --event_log_dir=./summaries \ # where to save training statistics
  --num_iterations=100000 \ # number of training iterations
  --sequence_length=10 \ # the number of total frames in a sequence
  --context_frames=2 \ # the number of ground truth frames to pass in at start
  --use_state=1 \ # whether or not to condition on actions and the initial state
  --num_masks=10 \ # the number of transformations and corresponding masks
  --schedsamp_k=900.0 \ # the constant used for scheduled sampling or -1
  --train_val_split=0.95 \ # the percentage of training data for validation
  --batch_size=32 \ # the training batch size
  --learning_rate=0.001 \ # the initial learning rate for the Adam optimizer
  #--pretrained_model=model \ # path to model to initialize from, random if emtpy
