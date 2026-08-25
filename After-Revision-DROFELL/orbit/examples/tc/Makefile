MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT ?= $(abspath $(MAKEFILE_DIR)/../../..)
PYTHON ?= python3
export PYTHONPATH := $(REPO_ROOT)$(if $(strip $(PYTHONPATH)),:$(PYTHONPATH))

BRIDGE?=br1
SUBNET?=10.200.1
NSPREFIX?=ns1-
LAT?=50ms
BW?=100mbit
NET:=BRIDGE=$(BRIDGE) SUBNET=$(SUBNET) NSPREFIX=$(NSPREFIX) LAT=$(LAT) BW=$(BW)
NC := $(NET) $(MAKEFILE_DIR)net_cluster
USE_NC ?= 1

info:
	$(NC) info 

upnet:
	$(NC) up

downnet:
	$(NC) down

reconfig:
	$(NC) reconfig

FIX_PATH := env PYTHONPATH="$(PYTHONPATH)"

ORBIT_LOCAL := --orbit=127.0.0.1:7241
ORBIT_NET   := --orbit=$(SUBNET).100:7241


ifeq ($(USE_NC),1)
define RUN_PREFIX
ID=$(1) $(NC) run $(FIX_PATH)
endef
ORBIT_ARG  := $(ORBIT_NET)
else
define RUN_PREFIX

endef
ORBIT_ARG  := $(ORBIT_LOCAL)
endif
PARSE_IDX = $(patsubst client%,%,$(1))
define RUN_NODE
	$(call RUN_PREFIX,$(call PARSE_IDX,$@)) \
	$(PYTHON) $(MAKEFILE_DIR)aux.py $(ORBIT_ARG) --host-id $(1) --name $(2)
endef

.PHONY: upnet orbit client1 client2 client3

orbit:
	$(call RUN_PREFIX,100) $(PYTHON) $(MAKEFILE_DIR)aux.py --host-id Orbit

client1:
	$(call RUN_NODE,node1,client1)

client2:
	$(call RUN_NODE,node2,client2)

client3:
	$(call RUN_NODE,node3,client3)
