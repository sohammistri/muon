#!/usr/bin/env python3
"""
Periodic sync of GPT-2 training artefacts to Azure Blob Storage.

Logs  : bbb cpr /root/code/muon/logs  az://.../training_logs/   (full copy every round)
Ckpts : diff-based — only new files are uploaded via bbb cp

Usage
-----
# Run as a background daemon (syncs every 30 min)
python GPT2/sync_cloud.py &

# Run once and exit (useful for testing)
python GPT2/sync_cloud.py --once

# Custom interval (seconds)
python GPT2/sync_cloud.py --interval 900
"""

import argparse
import glob
import logging
import os
import signal
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOCAL_LOGS = "/root/code/muon/logs"
CHECKPOINT_BASE = "/root/code/muon/GPT2"
REMOTE_LOGS = "az://m365transfer/data/mistrisoham/personal/muon/training_logs/"
REMOTE_CKPT_BASE = "az://m365transfer/data/mistrisoham/personal/muon/checkpoints/"

DEFAULT_INTERVAL = 30 * 60  # 30 minutes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sync_cloud")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr)."""
    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def remote_basenames(remote_dir: str) -> set[str]:
    """
    List files in a remote az:// directory and return a set of bare filenames.
    Returns an empty set if the directory doesn't exist yet or on any error.
    """
    rc, stdout, stderr = run_cmd(["bbb", "ls", remote_dir])
    if rc != 0:
        # Remote dir likely doesn't exist yet — treat as empty
        log.info("  bbb ls %s failed (probably new dir): %s", remote_dir, stderr.strip())
        return set()
    basenames = set()
    for line in stdout.splitlines():
        line = line.strip()
        if line:
            # Each line is a full az:// path — extract the basename
            basenames.add(line.rsplit("/", 1)[-1])
    return basenames


# ---------------------------------------------------------------------------
# Sync functions
# ---------------------------------------------------------------------------
def sync_logs() -> None:
    """Copy the entire logs directory to the cloud (idempotent recursive copy)."""
    if not os.path.isdir(LOCAL_LOGS):
        log.info("Logs dir %s does not exist yet — skipping.", LOCAL_LOGS)
        return

    log.info("Syncing logs: %s → %s", LOCAL_LOGS, REMOTE_LOGS)
    rc, stdout, stderr = run_cmd(["bbb", "cpr", LOCAL_LOGS, REMOTE_LOGS])
    if rc != 0:
        log.error("  logs sync FAILED (rc=%d): %s", rc, stderr.strip())
    else:
        log.info("  logs sync OK")
        if stdout.strip():
            log.debug("  stdout: %s", stdout.strip())


def sync_checkpoints() -> None:
    """
    For each d24-* checkpoint folder, compute the diff against the remote and
    upload only new files using bbb cp.
    """
    pattern = os.path.join(CHECKPOINT_BASE, "d24-*")
    ckpt_dirs = sorted(glob.glob(pattern))

    if not ckpt_dirs:
        log.info("No d24-* checkpoint dirs found under %s — skipping.", CHECKPOINT_BASE)
        return

    for local_dir in ckpt_dirs:
        folder_name = os.path.basename(local_dir)
        remote_dir = REMOTE_CKPT_BASE + folder_name + "/"
        log.info("Checkpoint dir: %s", folder_name)

        # Local files
        try:
            local_files = set(os.listdir(local_dir))
        except OSError as e:
            log.error("  Cannot list %s: %s — skipping.", local_dir, e)
            continue

        if not local_files:
            log.info("  Empty local dir — skipping.")
            continue

        # Remote files (may be empty if dir is new)
        existing = remote_basenames(remote_dir)
        new_files = sorted(local_files - existing)

        log.info(
            "  local=%d  remote=%d  new=%d",
            len(local_files), len(existing), len(new_files),
        )

        if not new_files:
            log.info("  All files already uploaded — nothing to do.")
            continue

        for fname in new_files:
            local_path = os.path.join(local_dir, fname)
            remote_path = remote_dir + fname
            rc, _, stderr = run_cmd(["bbb", "cp", local_path, remote_path])
            if rc != 0:
                log.error("  FAILED to upload %s: %s", fname, stderr.strip())
            else:
                log.info("  Uploaded %s", fname)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _shutdown
    log.info("Signal %d received — will exit after this round.", signum)
    _shutdown = True


def run_sync_round() -> None:
    log.info("=" * 60)
    log.info("Sync round started")
    log.info("=" * 60)
    sync_logs()
    sync_checkpoints()
    log.info("Sync round complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync GPT-2 artefacts to Azure Blob Storage.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sync and exit (no loop).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        metavar="SECONDS",
        help=f"Seconds between sync rounds (default: {DEFAULT_INTERVAL}).",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if args.once:
        run_sync_round()
        return

    log.info("Starting sync daemon (interval=%ds).  PID=%d", args.interval, os.getpid())
    log.info("Press Ctrl-C or send SIGTERM to stop after the current round.")

    while not _shutdown:
        run_sync_round()
        if _shutdown:
            break
        log.info("Sleeping %d s until next round…", args.interval)
        # Sleep in short chunks so SIGINT is responsive
        for _ in range(args.interval):
            if _shutdown:
                break
            time.sleep(1)

    log.info("Sync daemon exiting cleanly.")


if __name__ == "__main__":
    main()
