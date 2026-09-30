"""Run seeded rollouts for every prompt in prompts.yaml against a running
openpi policy server, logging per-episode success from LIBERO's own goal
check and saving per-episode videos.

The control loop mirrors openpi's examples/libero/main.py (same image
preprocessing, same state vector, same replan cadence) so in-distribution
numbers are comparable to openpi's published eval. openpi itself is not
modified; this script only uses the openpi-client package and LIBERO.

Fixed scene: episode k of EVERY prompt starts from LIBERO's precomputed
initial state k of the anchor task (all libero_goal tasks share one scene,
and the custom novel-goal BDDL files copy that scene verbatim).

Resumable: (prompt_id, episode) pairs already present in the log are
skipped, so an interrupted run continues where it left off.

Usage (client venv, with the policy server already running):
  python scripts/run_rollouts.py                # everything in prompts.yaml
  python scripts/run_rollouts.py --only id_bowl_plate,nv_cheese_plate
  python scripts/run_rollouts.py --episodes 3   # quick smoke run
"""

import argparse
import collections
import datetime
import json
import logging
import math
import pathlib
import time

import imageio
import numpy as np
import yaml

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render pi05_libero training data
DISPLAY_RESOLUTION = 512  # extra render for the demo video (does not touch the model input)
EPISODE_VIDEO_FPS = 20  # LIBERO control runs at 20 Hz


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils (same as openpi's LIBERO example)."""
    quat = np.array(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def resolve_bddl(spec_entry):
    kind, _, name = spec_entry.partition(":")
    if kind == "suite":
        return pathlib.Path(get_libero_path("bddl_files")) / "libero_goal" / (name + ".bddl")
    if kind == "custom":
        return REPO_ROOT / "bddl" / (name + ".bddl")
    raise ValueError("bad bddl spec: " + spec_entry)


def anchor_init_states(spec):
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[spec["task_suite"]]()
    for task_id in range(suite.n_tasks):
        if suite.get_task(task_id).name.endswith(spec["anchor_task"]):
            return suite.get_task_init_states(task_id)
    raise ValueError("anchor task not found: " + spec["anchor_task"])


def make_env(bddl_path, env_seed):
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(env_seed)  # affects object placement even with explicit init states
    return env


def display_frame(env, obs):
    """A higher-resolution agentview render for the video; falls back to the
    (rotated) model-input image if the extra render is unavailable."""
    try:
        frame = env.sim.render(
            width=DISPLAY_RESOLUTION, height=DISPLAY_RESOLUTION, camera_name="agentview"
        )
        return np.ascontiguousarray(frame[::-1])  # mujoco offscreen render is upside down
    except Exception:
        return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def run_episode(env, client, prompt, episode_idx, init_state, spec, video_path):
    env.reset()
    obs = env.set_init_state(init_state)
    action_plan = collections.deque()
    replan_steps = spec["replan_steps"]
    resize_size = spec["resize_size"]
    num_steps_wait = spec["num_steps_wait"]
    max_steps = prompt.get("max_steps", spec["max_steps_default"])

    frames = []
    success = False
    t = 0
    steps_taken = 0
    while t < max_steps + num_steps_wait:
        # Let objects settle after the initial state is applied (openpi does the same).
        if t < num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        # Preprocess exactly as openpi's LIBERO example: rotate 180 degrees to
        # match training, resize with padding to 224.
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
        )

        frames.append(display_frame(env, obs))

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ),
                "prompt": str(prompt["text"]),
            }
            action_chunk = client.infer(element)["actions"]
            assert len(action_chunk) >= replan_steps, (
                "policy predicts fewer steps than the replan cadence"
            )
            action_plan.extend(action_chunk[:replan_steps])

        action = action_plan.popleft()
        obs, _, done, _ = env.step(np.asarray(action).tolist())
        t += 1
        steps_taken += 1
        if done:  # LIBERO sets done from its own BDDL goal check
            success = True
            frames.append(display_frame(env, obs))
            break

    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(video_path), [np.asarray(f) for f in frames], fps=EPISODE_VIDEO_FPS)
    return success, steps_taken, len(frames)


def load_done(log_path):
    done = set()
    if log_path.exists():
        with open(str(log_path)) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["prompt_id"], rec["episode"]))
    return done


def write_summary(log_path, spec, out_dir):
    records = []
    with open(str(log_path)) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    by_prompt = collections.OrderedDict()
    for prompt in spec["prompts"]:
        eps = sorted(
            (r for r in records if r["prompt_id"] == prompt["id"]),
            key=lambda r: r["episode"],
        )
        if not eps:
            continue
        by_prompt[prompt["id"]] = {
            "text": prompt["text"],
            "group": prompt["group"],
            "kind": prompt["kind"],
            "num_episodes": len(eps),
            "num_successes": sum(1 for r in eps if r["success"]),
            "success_rate": sum(1 for r in eps if r["success"]) / float(len(eps)),
            "episodes": [
                {
                    "episode": r["episode"],
                    "success": r["success"],
                    "steps": r["steps"],
                    "video": r["video"],
                }
                for r in eps
            ],
        }
    summary = {
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "simulator": "LIBERO (libero_goal scene)",
        "anchor_task": spec["anchor_task"],
        "env_seed": spec["env_seed"],
        "generated_at": datetime.datetime.now().isoformat(),
        "prompts": by_prompt,
    }
    summary_path = out_dir / "summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", default=str(REPO_ROOT / "results"))
    parser.add_argument("--episodes", type=int, default=None, help="override prompts.yaml num_episodes")
    parser.add_argument("--only", default=None, help="comma-separated prompt ids to run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    spec = yaml.safe_load((REPO_ROOT / "prompts.yaml").read_text())
    num_episodes = args.episodes or spec["num_episodes"]
    only = set(args.only.split(",")) if args.only else None

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    done = load_done(log_path)
    if done:
        logging.info("resuming: %d episodes already logged", len(done))

    init_states = anchor_init_states(spec)
    assert len(init_states) >= num_episodes, "not enough precomputed init states"

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("connected to policy server: %s", client.get_server_metadata())

    log_file = open(str(log_path), "a")
    for prompt in spec["prompts"]:
        if only and prompt["id"] not in only:
            continue
        pending = [ep for ep in range(num_episodes) if (prompt["id"], ep) not in done]
        if not pending:
            logging.info("[%s] already complete", prompt["id"])
            continue

        bddl_path = resolve_bddl(prompt["bddl"])
        logging.info("[%s] %r (%s), bddl=%s", prompt["id"], prompt["text"], prompt["group"], bddl_path)
        env = make_env(bddl_path, spec["env_seed"])
        try:
            for ep in pending:
                video_path = out_dir / "episodes" / prompt["id"] / ("ep{:02d}.mp4".format(ep))
                start = time.time()
                success, steps, num_frames = run_episode(
                    env, client, prompt, ep, init_states[ep], spec, video_path
                )
                record = {
                    "prompt_id": prompt["id"],
                    "text": prompt["text"],
                    "group": prompt["group"],
                    "episode": ep,
                    "init_state_index": ep,
                    "env_seed": spec["env_seed"],
                    "success": bool(success),
                    "steps": steps,
                    "frames": num_frames,
                    "wall_time_s": round(time.time() - start, 1),
                    "video": str(video_path.relative_to(out_dir)),
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                logging.info(
                    "[%s] episode %d: %s (%d steps, %.0fs)",
                    prompt["id"], ep, "SUCCESS" if success else "failure", steps, record["wall_time_s"],
                )
        finally:
            env.close()

    log_file.close()
    summary_path, summary = write_summary(log_path, spec, out_dir)
    logging.info("summary written to %s", summary_path)
    for pid, entry in summary["prompts"].items():
        logging.info(
            "%-28s %-16s %d/%d", pid, entry["group"], entry["num_successes"], entry["num_episodes"]
        )


if __name__ == "__main__":
    main()
