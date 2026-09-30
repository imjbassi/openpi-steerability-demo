# One command per stage. Typical order:
#   make setup      (once; server + client environments)
#   make serve      (terminal 1; downloads the checkpoint on first run)
#   make rollouts   (terminal 2; all prompts x all episodes, resumable)
#   make video      (build media/demo.mp4 + media/shot1.gif from the logs)

# LIBERO is used straight from the openpi submodule checkout, so it must be
# on PYTHONPATH (openpi's LIBERO example does the same).
PY := PYTHONPATH=vendor/openpi/third_party/libero .venv-client/bin/python
MUJOCO_GL ?= egl

.PHONY: setup setup-openpi setup-client bddl check serve rollouts video smoke mock-server clean-results

setup: setup-openpi setup-client

setup-openpi:
	bash scripts/setup_openpi.sh

setup-client:
	bash scripts/setup_client.sh

# Regenerate the novel-goal BDDL files from LIBERO's own scene definition.
bddl:
	$(PY) scripts/gen_bddl.py

# Verify prompt-group claims against the 40 training task strings.
check:
	$(PY) scripts/check_prompts.py

serve:
	bash scripts/serve.sh

rollouts: bddl check
	MUJOCO_GL=$(MUJOCO_GL) $(PY) scripts/run_rollouts.py $(ARGS)

video:
	$(PY) scripts/make_video.py $(ARGS)

# Pipeline test without a GPU: mock policy server + 2 episodes per prompt.
mock-server:
	$(PY) scripts/mock_server.py

smoke: bddl check
	MUJOCO_GL=$(MUJOCO_GL) $(PY) scripts/run_rollouts.py --episodes 2 --out results-smoke

clean-results:
	rm -rf results results-smoke
