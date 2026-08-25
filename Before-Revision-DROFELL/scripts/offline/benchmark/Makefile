
infer-70B-1-client:
	SEQS=1 python inference/benchmark.py --profile 70B --output inference/results/70B-1
infer-70B-3-client:
	SEQS=3 python inference/benchmark.py --profile 70B --output inference/results/70B-3
infer-70B-8-client:
	SEQS=8 python inference/benchmark.py --profile 70B --output inference/results/70B-8
infer-70B-16-client:
	SEQS=16 python inference/benchmark.py --profile 70B --output inference/results/70B-16

infers: infer-70B-1-client infer-70B-3-client infer-70B-8-client infer-70B-16-client

train-70B-1-client-1-batch:
	INSTANCES=1 MINIBATCH_SIZE=1 BATCH_SIZE=1 python train/benchmark.py --profile 70B --output train/results/70B-1-1

train-70B-1-client-16-batch:
	INSTANCES=1 MINIBATCH_SIZE=4 BATCH_SIZE=4 python train/benchmark.py --profile 70B --output train/results/70B-1-16

train-70B-1-client-64-batch:
	INSTANCES=1 MINIBATCH_SIZE=4 BATCH_SIZE=16 python train/benchmark.py --profile 70B --output train/results/70B-1-64

train-70B-3-client-1-batch:
	INSTANCES=3 MINIBATCH_SIZE=1 BATCH_SIZE=1 python train/benchmark.py --profile 70B --output train/results/70B-3-1

train-70B-3-client-16-batch:
	INSTANCES=3 MINIBATCH_SIZE=4 BATCH_SIZE=4 python train/benchmark.py --profile 70B --output train/results/70B-3-16

train-70B-3-client-64-batch:
	INSTANCES=3 MINIBATCH_SIZE=4 BATCH_SIZE=16 python train/benchmark.py --profile 70B --output train/results/70B-3-64

train-70B-8-client-1-batch:
	INSTANCES=8 MINIBATCH_SIZE=1 BATCH_SIZE=1 python train/benchmark.py --profile 70B --output train/results/70B-8-1

train-70B-8-client-16-batch:
	INSTANCES=8 MINIBATCH_SIZE=4 BATCH_SIZE=4 python train/benchmark.py --profile 70B --output train/results/70B-8-16

train-70B-8-client-64-batch:
	INSTANCES=8 MINIBATCH_SIZE=4 BATCH_SIZE=16 python train/benchmark.py --profile 70B --output train/results/70B-8-64

train-70B-16-client-1-batch:
	INSTANCES=16 MINIBATCH_SIZE=1 BATCH_SIZE=1 python train/benchmark.py --profile 70B --output train/results/70B-16-1

train-70B-16-client-16-batch:
	INSTANCES=16 MINIBATCH_SIZE=4 BATCH_SIZE=4 python train/benchmark.py --profile 70B --output train/results/70B-16-16

train-70B-16-client-64-batch:
	INSTANCES=16 MINIBATCH_SIZE=4 BATCH_SIZE=16 python train/benchmark.py --profile 70B --output train/results/70B-16-64

trains: train-70B-1-client-1-batch train-70B-1-client-16-batch train-70B-1-client-64-batch train-70B-3-client-1-batch train-70B-3-client-16-batch train-70B-3-client-64-batch train-70B-8-client-1-batch train-70B-8-client-16-batch train-70B-8-client-64-batch train-70B-16-client-1-batch train-70B-16-client-16-batch train-70B-16-client-64-batch