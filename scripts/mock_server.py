"""A stand-in policy server speaking the openpi-client websocket protocol.

Lets the rollout and video pipeline be exercised end to end on a machine
without a GPU (episodes will simply fail their goal checks). It returns
small random action chunks with the same shape the real server produces.

Usage (client venv):  python scripts/mock_server.py --port 8000
"""

import argparse
import logging

import numpy as np
import websockets.sync.server

from openpi_client import msgpack_numpy

ACTION_DIM = 7
ACTION_HORIZON = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    rng = np.random.RandomState(args.seed)
    packer = msgpack_numpy.Packer()

    def handler(conn):
        logging.info("client connected")
        conn.send(packer.pack({"mock": True}))
        while True:
            try:
                obs = msgpack_numpy.unpackb(conn.recv())
            except Exception:
                break
            assert "prompt" in obs and "observation/image" in obs
            actions = rng.uniform(-0.2, 0.2, size=(ACTION_HORIZON, ACTION_DIM))
            actions[:, -1] = -1.0
            conn.send(packer.pack({"actions": actions}))
        logging.info("client disconnected")

    with websockets.sync.server.serve(
        handler, "0.0.0.0", args.port, compression=None, max_size=None
    ) as server:
        logging.info("mock policy server on port %d", args.port)
        server.serve_forever()


if __name__ == "__main__":
    main()
