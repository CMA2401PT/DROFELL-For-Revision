PY := /opt/conda/bin/python
PKGPATH := ~/projects/LLM/island
export PYTHONPATH := $(PKGPATH)

BRIDGE?=br1
SUBNET?=10.200.1
NSPREFIX?=ns1-
LAT?=0ms
BW?=10000mbit
NET:=BRIDGE=$(BRIDGE) SUBNET=$(SUBNET) NSPREFIX=$(NSPREFIX) LAT=$(LAT) BW=$(BW)
NC := $(NET) ./net_cluster
USE_NC ?= 0

info:
	$(NC) info 

upnet:
	$(NC) up

downnet:
	$(NC) down

reconfig:
	$(NC) reconfig

FIX_PATH := env PYTHONPATH="$(PKGPATH)"
ORBIT_LOCAL := 127.0.0.1:7241
ORBIT_NET   := $(SUBNET).100:7241
OUTPUT ?="results/test"

ifeq ($(USE_NC),1)
define RUN_PREFIX
ID=$(1) $(NC) run $(FIX_PATH) $(PY) 
endef
ORBIT  := $(ORBIT_NET)
else
define RUN_PREFIX
$(FIX_PATH) $(PY) 
endef
ORBIT  := $(ORBIT_LOCAL)
endif

.PHONY: upnet orbit client1 client2 client3

# orbit-7B:
# 	$(call RUN_PREFIX,100) setups/7B_drop.py --host-id Orbit

# dealer-7B-1-1:
# 	$(call RUN_PREFIX,1) setups/7B_drop.py \
# 		--host-id=satellite-1 \
# 		--orbit=$(ORBIT) \
# 		--name=T1-dealer-7B-1 \
# 		--num-decoders=32 \
# 		--device=cuda:0

# dealer-7B-2-1:
# 	$(call RUN_PREFIX,1) setups/7B_drop.py \
# 		--host-id=satellite-1 \
# 		--orbit=$(ORBIT) \
# 		--name=T1-dealer-7B-2-1 \
# 		--num-decoders=16 \
# 		--device=cuda:0

# dealer-7B-4-1:
# 	$(call RUN_PREFIX,1) setups/7B_drop.py \
# 		--host-id=satellite-1 \
# 		--orbit=$(ORBIT) \
# 		--name=T1-dealer-7B-2-1 \
# 		--num-decoders=8 \
# 		--device=cuda:0

# dealer-7B-2-2:
# 	$(call RUN_PREFIX,2) setups/7B_drop.py \
# 		--host-id=satellite-2 \
# 		--orbit=$(ORBIT) \
# 		--name=T1-dealer-7B-2-2 \
# 		--num-decoders=16 \
# 		--device=cuda:0

# bench-7B-infer:
# 	$(call RUN_PREFIX,10) \
# 		benchmark_infer.py \
# 		--orbit=$(ORBIT) \
# 		--profile=7B \
# 		--output="results/inference/test"

# bench-7B-train:
# 	$(call RUN_PREFIX,10) \
# 		benchmark_train.py \
# 		--orbit=$(ORBIT) \
# 		--profile=7B \
# 		--output="results/inference/test"

setup-7B-drop:
	$(FIX_PATH) $(PY) setup.py --script 7B_drop

bench-7B:
	$(FIX_PATH) $(PY) benchmark_infer.py \
		--orbit=$(ORBIT) \
		--profile=7B \
		--output=$(OUTPUT)

bench-7B-drop:
	$(FIX_PATH) $(PY) benchmark_infer.py \
		--orbit=$(ORBIT) \
		--profile=7B_drop \
		--output=$(OUTPUT)

orbit-70B-infer:
# 	SPLITS="40,10,8,6,4" will slightly reduce memory consumption than default SPLITS="10,8,6,4"
	SPLITS="40,10,8,6,4" $(call RUN_PREFIX,100) setups/70B.py --host-id Orbit

dealer-70B-2-1-infer:
	$(call RUN_PREFIX,1) setups/70B.py \
		--host-id=satellite-1 \
		--orbit=$(ORBIT) \
		--name=T1-dealer-70B-2-1 \
		--num-decoders=40 \
		--device=cuda:0 \
		--task-type=infer

dealer-70B-2-2-infer:
	$(call RUN_PREFIX,2) setups/70B.py \
		--host-id=satellite-2 \
		--orbit=$(ORBIT) \
		--name=T1-dealer-70B-2-2 \
		--num-decoders=40 \
		--device=cuda:1 \
		--task-type=infer

bench-70B-infer:
	$(call RUN_PREFIX,10) \
		benchmark_infer.py \
		--orbit=$(ORBIT) \
		--profile=70B \
		--output=$(OUTPUT)



orbit-70B-train:
	SPLITS="40,10,8,6,4" $(call RUN_PREFIX,100) setups/70B.py --host-id Orbit

dealer-70B-2-1-train:
	$(call RUN_PREFIX,1) setups/70B.py \
		--host-id=satellite-1 \
		--orbit=$(ORBIT) \
		--name=T1-dealer-70B-2-1 \
		--num-decoders=40 \
		--device=cuda:0 \
		--task-type=train

dealer-70B-2-2-train:
	$(call RUN_PREFIX,2) setups/70B.py \
		--host-id=satellite-2 \
		--orbit=$(ORBIT) \
		--name=T1-dealer-70B-2-2 \
		--num-decoders=40 \
		--device=cuda:1 \
		--task-type=train

bench-70B-train:
	$(call RUN_PREFIX,10) \
		benchmark_train.py \
		--orbit=$(ORBIT) \
		--profile=70B \
		--output=$(OUTPUT)

# DEVIDES=1 DEVICES=1 DUPLICATES=1 make setup-7B-drop group=1
# DROP_PROB=0       POSTPONE=0   MULTI_PATH=1 OUTPUT="results/bench-drop/tmp/A-0" make bench-7B-drop &&\
# DROP_PROB=0.0001  POSTPONE=0   MULTI_PATH=1 OUTPUT="results/bench-drop/tmp/B-0" make bench-7B-drop &&\
# DROP_PROB=0.001   POSTPONE=0   MULTI_PATH=1 OUTPUT="results/bench-drop/tmp/C-0" make bench-7B-drop &&\
# DROP_PROB=0.01    POSTPONE=0   MULTI_PATH=1 OUTPUT="results/bench-drop/tmp/D-0" make bench-7B-drop &&\
# DROP_PROB=0.05    POSTPONE=0   MULTI_PATH=1 OUTPUT="results/bench-drop/tmp/E-0" make bench-7B-drop &&\
# DROP_PROB=0       POSTPONE=0.3 MULTI_PATH=1 OUTPUT="results/bench-drop/tmp/A-1" make bench-7B-drop &&\
# DROP_PROB=0.0001  POSTPONE=0.3 MULTI_PATH=1 OUTPUT="results/bench-drop/tmp/B-1" make bench-7B-drop &&\
# DROP_PROB=0.001   POSTPONE=0.3 MULTI_PATH=1 OUTPUT="results/bench-drop/tmp/C-1" make bench-7B-drop &&\
# DROP_PROB=0.01    POSTPONE=0.3 MULTI_PATH=1 OUTPUT="results/bench-drop/tmp/D-1" make bench-7B-drop &&\
# DROP_PROB=0.05    POSTPONE=0.3 MULTI_PATH=1 OUTPUT="results/bench-drop/tmp/E-1" make bench-7B-drop &&\
# cp -r results/bench-drop/tmp  results/bench-drop/1 && rm -r  results/bench-drop/tmp

# DEVIDES=1 DEVICES=1 DUPLICATES=1 make setup-7B-drop group=1
# cp -r results/bench-drop/tmp  results/bench-drop/1 && rm -r  results/bench-drop/tmp
# DEVIDES=2 DEVICES=1 DUPLICATES=1 make setup-7B-drop group=2
# cp -r results/bench-drop/tmp  results/bench-drop/2 && rm -r  results/bench-drop/tmp
# DEVIDES=4 DEVICES=1 DUPLICATES=1 make setup-7B-drop group=3
# cp -r results/bench-drop/tmp  results/bench-drop/3 && rm -r  results/bench-drop/tmp
# DEVIDES=2 DEVICES=2 DUPLICATES=1 make setup-7B-drop group=4
# cp -r results/bench-drop/tmp  results/bench-drop/4 && rm -r  results/bench-drop/tmp
# DEVIDES=4 DEVICES=2 DUPLICATES=1 make setup-7B-drop group=5
# cp -r results/bench-drop/tmp  results/bench-drop/5 && rm -r  results/bench-drop/tmp

# DEVIDES=1 DEVICES=2 DUPLICATES=2 make setup-7B-drop group=6
# cp -r results/bench-drop/tmp  results/bench-drop/6 && rm -r  results/bench-drop/tmp
# DEVIDES=2 DEVICES=2 DUPLICATES=2 make setup-7B-drop group=7
# cp -r results/bench-drop/tmp  results/bench-drop/7 && rm -r  results/bench-drop/tmp
# DEVIDES=4 DEVICES=2 DUPLICATES=2 make setup-7B-drop group=8
# cp -r results/bench-drop/tmp  results/bench-drop/8 && rm -r  results/bench-drop/tmp
# DROP_PROB=0       POSTPONE=0   MULTI_PATH=2 OUTPUT="results/bench-drop/tmp/A-0" make bench-7B-drop &&\
# DROP_PROB=0.0001  POSTPONE=0   MULTI_PATH=2 OUTPUT="results/bench-drop/tmp/B-0" make bench-7B-drop &&\
# DROP_PROB=0.001   POSTPONE=0   MULTI_PATH=2 OUTPUT="results/bench-drop/tmp/C-0" make bench-7B-drop &&\
# DROP_PROB=0.01    POSTPONE=0   MULTI_PATH=2 OUTPUT="results/bench-drop/tmp/D-0" make bench-7B-drop &&\
# DROP_PROB=0.05    POSTPONE=0   MULTI_PATH=2 OUTPUT="results/bench-drop/tmp/E-0" make bench-7B-drop &&\
# DROP_PROB=0       POSTPONE=0.3 MULTI_PATH=2 OUTPUT="results/bench-drop/tmp/A-1" make bench-7B-drop &&\
# DROP_PROB=0.0001  POSTPONE=0.3 MULTI_PATH=2 OUTPUT="results/bench-drop/tmp/B-1" make bench-7B-drop &&\
# DROP_PROB=0.001   POSTPONE=0.3 MULTI_PATH=2 OUTPUT="results/bench-drop/tmp/C-1" make bench-7B-drop &&\
# DROP_PROB=0.01    POSTPONE=0.3 MULTI_PATH=2 OUTPUT="results/bench-drop/tmp/D-1" make bench-7B-drop &&\
# DROP_PROB=0.05    POSTPONE=0.3 MULTI_PATH=2 OUTPUT="results/bench-drop/tmp/E-1" make bench-7B-drop &&\
# cp -r results/bench-drop/tmp  results/bench-drop/8 && rm -r  results/bench-drop/tmp

# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit SEQS=1 OUTPUT=results/bench-infer/1 make bench-70B-infer &&\
# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit SEQS=3 OUTPUT=results/bench-infer/4 make bench-70B-infer &&\
# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit SEQS=8 OUTPUT=results/bench-infer/7 make bench-70B-infer &&\
# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit SEQS=16 OUTPUT=results/bench-infer/10 make bench-70B-infer &&\
# LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit SEQS=1 OUTPUT=results/bench-infer/2 make bench-70B-infer &&\
# LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit SEQS=3 OUTPUT=results/bench-infer/5 make bench-70B-infer &&\
# LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit SEQS=8 OUTPUT=results/bench-infer/8 make bench-70B-infer &&\
# LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit SEQS=16 OUTPUT=results/bench-infer/11 make bench-70B-infer &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit SEQS=1 OUTPUT=results/bench-infer/3 make bench-70B-infer &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit SEQS=3 OUTPUT=results/bench-infer/6 make bench-70B-infer &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit SEQS=8 OUTPUT=results/bench-infer/9 make bench-70B-infer &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit SEQS=16 OUTPUT=results/bench-infer/12 make bench-70B-infer


# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=1  MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/1-A  make bench-70B-train &&\
# sleep 1 &&\
# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=1  MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/1-B  make bench-70B-train &&\
# sleep 1 &&\
# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=1  MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/1-C  make bench-70B-train &&\
# sleep 1 &&\
# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=3  MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/4-A  make bench-70B-train &&\
# sleep 1 &&\
# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=3  MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/4-B  make bench-70B-train &&\
# sleep 1 &&\
# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=3  MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/4-C  make bench-70B-train &&\
# sleep 1 &&\
# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=8  MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/7-A  make bench-70B-train &&\
# sleep 1 &&\
# LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=8  MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/7-B  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=8  MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/7-C  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=16 MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/10-A make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=16 MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/10-B make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=10000mbit make reconfig && LAT=0ms BW=10000mbit INSTANCES=16 MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/10-C make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=1  MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/2-A  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=1  MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/2-B  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=1  MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/2-C  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=3  MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/5-A  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=3  MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/5-B  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=3  MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/5-C  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=8  MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/8-A  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=8  MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/8-B  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=8  MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/8-C  make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=16 MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/11-A make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=16 MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/11-B make bench-70B-train &&\
# sleep 1 &&\
#LAT=0ms BW=100mbit make reconfig && LAT=0ms BW=100mbit INSTANCES=16 MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/11-C make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=1  MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/3-A  make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=1  MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/3-B  make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=1  MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/3-C  make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=3  MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/6-A  make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=3  MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/6-B  make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=3  MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/6-C  make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=8  MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/9-A  make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=8  MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/9-B  make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=8  MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/9-C  make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=16 MINIBATCH_SIZE=1 BATCH_SIZE=1  OUTPUT=results/bench-train/12-A make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=16 MINIBATCH_SIZE=4 BATCH_SIZE=4  OUTPUT=results/bench-train/12-B make bench-70B-train &&\
# sleep 1 &&\
# LAT=50ms BW=100mbit make reconfig && LAT=50ms BW=100mbit INSTANCES=16 MINIBATCH_SIZE=4 BATCH_SIZE=16 OUTPUT=results/bench-train/12-C make bench-70B-train