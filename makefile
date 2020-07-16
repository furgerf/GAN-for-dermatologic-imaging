VENV=venv
BIN=$(VENV)/bin
NICE=nice -n 19
SHELL=/bin/bash

ARGS=
EVAL_DIR=output
DATE:=$(shell date '+%Y%m%d_%H%M%S')
DESC=no-desc
EID=$(DESC)_$(DATE)
THIS_EVAL_DIR:=$(EVAL_DIR)/$(EID)
EVAL_LOG:=$(THIS_EVAL_DIR)/evaluation.log
PERCENT:=%
GPU=2

DIRECTORIES=$(EVAL_DIR)

.PHONY: eval evaluation hand-gen handpics-dcgan explore list-training-sets \
	create-patches test-segmentation create-movie lint list-docker wait pause resume kill \
	generate-samples find-good-sample interpolate-noise all-interpolations compare-models compare-training-samples \
	noise noise-test segmentation segmentation-test patho-addition-or-removal \
	patho-addition-or-removal-test cycle-without-patho cycle-without-patho-test cycle cycle-test \
	get-directory-hash get-output get-output-loop view-last-image tensorboard freeze install uninstall

eval:
	@# the user should specify the DESC of the evaluation which is used for the filename
	@# still, the user can specify an EID (e.g. foo) to overwrite the actual eid without changing anything else
	@mkdir -p $(THIS_EVAL_DIR)
	$(NICE) $(BIN)/python src/main.py --eid=$(EID) --config-file=$(DESC).json \
		$(ARGS) 2>&1 | tee $(EVAL_LOG); [ $${PIPESTATUS[0]} -eq 0 ]

evaluation:
ifndef MOD
	$(error Must specify model name)
endif
ifndef DATA
	$(error Must specify data directory)
endif
ifdef PID
	@$(MAKE) --no-print-directory wait
endif
	@mkdir -p $(THIS_EVAL_DIR)
	$(NICE) $(BIN)/python src/main.py --eid=$(EID) --model-name=$(MOD) --data-dir=$(DATA) \
		$(ARGS) 2>&1 | tee $(EVAL_LOG); [ $${PIPESTATUS[0]} -eq 0 ]

noise:
	@# noise to image evaluation
	@$(MAKE) --no-print-directory evaluation ARGS='--input-type=noise --target-type=image $(ARGS)'

noise-test:
	@$(MAKE) --no-print-directory noise EID=foo MOD=Simple256Convolution DATA=new-patches-256-more-patho-back \
		ARGS='--batch-size=2 $(ARGS)'

segmentation:
	@# image to image evaluation
	@$(MAKE) --no-print-directory evaluation ARGS='--input-type=image --target-type=patho $(ARGS)'

segmentation-test:
	@$(MAKE) --no-print-directory segmentation EID=foo MOD=FromImage256Convolution DATA=new-patches-256-more-patho-back \
		ARGS='--batch-size=2 $(ARGS)'

patho-addition-or-removal:
	@# two images to image evaluation
	@$(MAKE) --no-print-directory evaluation ARGS='--input-type=image --second-input-type=patho --target-type=image $(ARGS)'

patho-addition-or-removal-test:
	@$(MAKE) --no-print-directory patho-addition-or-removal EID=foo MOD=FromImage256Convolution DATA=new-patches-256-more-patho-back \
		ARGS='--target-data-dir=new-patches-256-no-patho-back --batch-size=2 $(ARGS)'

cycle-without-patho:
	@# two-way image to image evaluation
	@$(MAKE) --no-print-directory evaluation ARGS='--cycle --input-type=image --target-type=image $(ARGS)'

cycle-without-patho-test:
	@$(MAKE) --no-print-directory cycle-without-patho EID=foo MOD=FromImage256Convolution DATA=new-patches-256-more-patho-back \
		ARGS='--target-data-dir=new-patches-256-no-patho-back --batch-size=2 $(ARGS)'

cycle:
	@# two-way two images to image evaluation
	@$(MAKE) --no-print-directory patho-addition-or-removal ARGS='--cycle $(ARGS)'

cycle-test:
	@$(MAKE) --no-print-directory cycle EID=foo MOD=FromImage256Convolution DATA=new-patches-256-more-patho-back \
		ARGS='--target-data-dir=new-patches-256-no-patho-back --batch-size=2 $(ARGS)'

create-patches:
	@mkdir -p $(THIS_EVAL_DIR)
	$(NICE) $(BIN)/python src/create_patches.py --output-dir=$(THIS_EVAL_DIR) $(ARGS) 2>&1 \
		| tee $(EVAL_LOG); [ $${PIPESTATUS[0]} -eq 0 ]

test-segmentation:
ifndef MOD
	$(error Must specify model name)
endif
ifndef DIR
	$(error Must specify the eval directory)
endif
ifndef DATA
	$(error Must specify data directory)
endif
ifndef EPOCH
	$(error Must specify the epoch to load)
endif
	$(NICE) $(BIN)/python src/test_segmentation.py --eval-dir=$(DIR) --model-name=$(MOD) --test-data=$(DATA) --epoch=$(EPOCH) $(ARGS)

generate-samples:
ifndef MOD
	$(error Must specify model name)
endif
ifndef DIR
	$(error Must specify the eval directory)
endif
ifndef EPOCH
	$(error Must specify the epoch to load)
endif
	$(NICE) $(BIN)/python src/generate_samples.py --eval-dir=$(DIR) --model-name=$(MOD) --epoch=$(EPOCH) --description=$(DESC) $(ARGS)

find-good-sample:
ifndef MOD
	$(error Must specify model name)
endif
ifndef DIR
	$(error Must specify the eval directory)
endif
ifndef EPOCH
	$(error Must specify the epoch to load)
endif
	$(NICE) $(BIN)/python src/find_good_sample.py --eval-dir=$(DIR) --model-name=$(MOD) --epoch=$(EPOCH) --description=$(DESC) $(ARGS)

interpolate-noise:
ifndef MOD
	$(error Must specify model name)
endif
ifndef EID
	$(error Must specify the eval ID)
endif
ifndef EPOCH
	$(error Must specify the epoch to load)
endif
	$(NICE) $(BIN)/python src/interpolate_noise.py --eid=$(EID) --model-name=$(MOD) --epoch=$(EPOCH) $(ARGS)

all-interpolations:
ifndef MOD
	$(error Must specify model name)
endif
ifndef EID
	$(error Must specify the eval ID)
endif
ifndef EPOCH
	$(error Must specify the epoch to load)
endif
	$(NICE) $(BIN)/python src/interpolate_noise.py --eid=$(EID) --model-name=$(MOD) --epoch=$(EPOCH) --description=pairs $(ARGS)
	$(NICE) $(BIN)/python src/interpolate_noise.py --eid=$(EID) --model-name=$(MOD) --epoch=$(EPOCH) --description=pairs-spherical --spherical $(ARGS)
	$(NICE) $(BIN)/python src/interpolate_noise.py --eid=$(EID) --model-name=$(MOD) --epoch=$(EPOCH) --description=grid --grid $(ARGS)
	$(NICE) $(BIN)/python src/interpolate_noise.py --eid=$(EID) --model-name=$(MOD) --epoch=$(EPOCH) --description=grid-spherical --grid --spherical $(ARGS)
	$(NICE) $(BIN)/python src/interpolate_noise.py --eid=$(EID) --model-name=$(MOD) --epoch=$(EPOCH) --description=single --single $(ARGS)
	$(NICE) $(BIN)/python src/interpolate_noise.py --eid=$(EID) --model-name=$(MOD) --epoch=$(EPOCH) --description=single-spherical --single --spherical $(ARGS)

compare-models:
ifndef MOD
	$(error Must specify model name)
endif
ifndef DIR
	$(error Must specify the eval directory)
endif
ifndef EPOCH
	$(error Must specify the epoch to load)
endif
ifndef MOD_2
	$(error Must specify a second model name)
endif
ifndef DIR_2
	$(error Must specify the second eval directory)
endif
ifndef EPOCH_2
	$(error Must specify the second epoch to load)
endif
ifndef DESC
	$(error Must specify a description)
endif
ifndef DATA
	$(error Must specify data directory)
endif
	$(NICE) $(BIN)/python src/compare_model_output.py --description=$(DESC) --data-dir=$(DATA) \
		--eval-dir=$(DIR) --second-eval-dir=$(DIR_2) \
		--model-name=$(MOD) --second-model-name=$(MOD_2) --epoch=$(EPOCH) --second-epoch=$(EPOCH_2) $(ARGS)

create-movie:
ifndef DIR
	$(error Must specify the eval directory)
endif
	# $(NICE) ffmpeg -r 1 -pattern_type glob -i 'output/$(DIR)/figures/$(PREFIX)image_at_epoch_*.png' output/$(DIR)/movie-$$(basename $(DIR)).mp4
	$(NICE) ffmpeg -r 2 -i output/$(DIR)/figures/$(PREFIX)image_at_epoch_$(PERCENT)04d.png -vframes 100 output/$(DIR)/movie-$$(basename $(DIR)).mp4

get-output:
	rsync -avz --exclude=checkpoints --exclude=foo gpu0$(GPU):/home/iafurger/output/$(PAT)* output/ || true

get-output-loop:
	@while true; do \
		if ! ping -c1 -w1 10.177.1.254 > /dev/null; then echo "Skipping download: not conected"; sleep 10m; continue; fi; \
		time $(MAKE) --no-print-directory get-output; \
		echo "*** Downloaded output on $$(date) ***"; \
		sleep 5m; \
		done

get-directory-hash:
ifndef DIR
	$(error Must specify the directory)
endif
	@for dir in $(DIR); do \
		echo "$$dir: $$(find $$dir -type f -exec md5sum {} \; | sort | md5sum | cut -d' ' -f1)"; \
	done

view-last-image:
ifndef DESC
	$(error Must specify the description of the evaluation to view)
endif
	@last_image=$$(ssh gpu0$(GPU) -C 'ls -1 /home/iafurger/output/*$(DESC)*/figures | tail -n1'); \
	test -z $$last_image && exit 1; \
	tmpfile=$$(mktemp /tmp/remote-image.XXXXX); \
	rsync -avz gpu0$(GPU):/home/iafurger/output/*$(DESC)*/figures/$$last_image $$tmpfile; \
	geeqie $$tmpfile; \
	rm $$tmpfile

tensorboard:
ifndef DIR
	$(error Must specify the directory)
endif
	@DIRS=($(DIR)); \
	for ((i=0;i<$${#DIRS[@]};i++)); do \
		DIRS[i]="$$i:$$(echo output/*$${DIRS[i]}*)"; \
	done; \
	echo "Logdir(s): $$(echo "$${DIRS[@]}" | tr ' ' ,)"; \
	tensorboard --logdir="$$(echo "$${DIRS[@]}" | tr ' ' ,)"

hand-gen:
	@mkdir -p $(THIS_EVAL_DIR)
	$(NICE) $(BIN)/python src/hand-gen.py --eid=$(EID) $(ARGS) 2>&1 | tee $(EVAL_LOG); [ $${PIPESTATUS[0]} -eq 0 ]

handpics-dcgan:
	@mkdir -p $(THIS_EVAL_DIR)
	$(NICE) $(BIN)/python src/handpics-dcgan.py 2>&1 | tee $(EVAL_LOG); [ $${PIPESTATUS[0]} -eq 0 ]

explore:
	@PYTHONPATH="$$PYTHONPATH:src/" $(BIN)/ipython --no-banner --no-confirm-exit -i src/explore.py

list-training-sets:
	@find ./data/ -mindepth 1 -maxdepth 1 -type d -exec \
		sh -c '[ -d {}/images_orig -o -d {}/image ] && \
			echo "$$(basename {}): $$(find {}/images_orig/ {}/image/ -type f 2> /dev/null | egrep "png|jpg|jpeg" | wc -l) images" || \
			echo "* Skipping $$(basename {}) *"' \; | sort

lint:
	@PYTHONPATH="$$PYTHONPATH:src/" $(BIN)/pylint src/*.py --ignore=venv/ --ignored-modules=tensorflow -f colorized -r n \
		--msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}"

list-docker:
	$(info Columns: PID, start, command)
	@ps u | grep [d]ocker-compose | grep iafurger | tr -s ' ' | cut -d' ' -f 2,9,11-

wait:
ifdef PID
	$(info Waiting for process $(PID)...)
	@while [ -d /proc/$$PID ]; do \
		sleep 60; \
	done
endif

pause:
ifndef PID
	$(error Must specify pid)
endif
	kill -STOP $(PID) $$(ps -o pid= --ppid $(PID))

resume:
ifndef PID
	$(error Must specify pid)
endif
	kill -CONT $(PID) $$(ps -o pid= --ppid $(PID))

kill:
ifndef PID
	$(error Must specify pid)
endif
	kill -TERM $(PID) $$(ps -o pid= --ppid $(PID))

venv:
	@[ $$(command -v python3) ] || (echo 'python 3 is missing'; exit 1)
	@(( $$(readlink $$(which python3) | cut -d. -f2) >= 5 )) || (echo 'python >= 3.5 is required'; exit 1)
	virtualenv -p python3 $(VENV)

freeze: venv
	$(BIN)/pip freeze | grep -v "pkg-resources" > requirements.txt

install: venv
	$(BIN)/pip install -r requirements.txt
	mkdir -p $(DIRECTORIES)

uninstall:
	rm -rf $(VENV)

