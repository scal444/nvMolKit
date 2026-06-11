---
name: remote-gpu-node
description: >-
  Connect to a remote GPU compute node via SSH jump host and Docker container
  for building, testing, benchmarking, and running CUDA workloads. Use when the
  user wants to run something on a remote GPU node, sync code to a cluster,
  execute commands in a Docker container on a compute node, or references
  computelab/cluster access. This skill is a prerequisite for build/test skills
  that need remote GPU hardware.
---

# Remote GPU Node Access

Imported from `~/.cursor/skills/remote-gpu-node/SKILL.md`, with Codex shell
permission conventions folded in.

Provides SSH+Docker access to GPU compute nodes behind a jump host. Other
skills chain onto this one for build, test, and benchmark workflows.

## Step 0: Ask For The Compute Node Up Front

The compute node hostname is session-specific and is essentially never the same
as last time. Before doing any sync/build/run work, ask the user which compute
node to use, for example:

```text
What compute node is allocated for this session?
```

Do not assume any cached default for `COMPUTE_NODE`. The jump host and rsync
destination prefix tend to stay stable for this user, so those can keep their
defaults until corrected.

## Connection Parameters

Set these before use. Defaults reflect kboyd's typical environment:

| Variable | Default | Purpose |
|---|---|---|
| `JUMP_HOST` | `kboyd@computelab-sc-01.nvidia.com` | SSH jump/bastion host |
| `COMPUTE_NODE` | ask the user every session | Target GPU node, e.g. `kboyd@luna-prod-78-80gb` |
| `REMOTE_DEST` | `kboyd@computelab.nvidia.com:/home/scratch.kboyd_other/` | rsync destination prefix |
| `LOCAL_SRC` | workspace root | Local directory to sync |

For nvMolKit work, the standing mapping is:

| Host scratch path | Container path | Sync shape |
|---|---|---|
| `/home/scratch.kboyd_other/nvmolkit/` | `/nvmolkit/` | Sync worktree contents into `${REMOTE_DEST}nvmolkit/` |

Use this default for nvMolKit unless the user explicitly gives a different
path. The compute node is still session-specific and must be requested every
session.

Export these once at the top of the session and reuse:

```bash
export JUMP_HOST="kboyd@computelab-sc-01.nvidia.com"
export COMPUTE_NODE="kboyd@<host-from-user>"
export REMOTE_DEST="kboyd@computelab.nvidia.com:/home/scratch.kboyd_other/"
export LOCAL_SRC="/path/to/workspace/"
```

## Step 1: Find A Sync Target

The container has NFS mounts, such as `/nvmolkit` and `/data`, that map to
specific subdirectories on the host's `/home/scratch.<user>_other/` tree. Some
of those mounts may be manually managed by the user. Before assuming a mount is
fair game, ask:

```text
Which path on the remote scratch should I sync to?
```

Confirm the answer maps the host-side rsync destination, `REMOTE_DEST` plus the
chosen subdirectory, to a container-visible mount. Do not pick a directory the
user is actively syncing themselves.

`/tmp` inside the container is container-local only. Rsync from the local
machine hits the host filesystem, not the container's `/tmp`. Use a host scratch
path that is mounted into the container.

## Primitives

### Sync Local Tree To Remote

```bash
rsync -uavz --chmod=ugo=rwX --delete \
  --exclude="*__pycache__*" --exclude=".idea*" --exclude="*egg-info*" \
  --exclude=.git --exclude="cmake-*" --exclude="build" --exclude="build_*" \
  --exclude="_deps" --exclude="build-*" \
  "$LOCAL_SRC" "$REMOTE_DEST<subdir>/"
```

For the standard nvMolKit mount, use a trailing slash on the source so rsync
replaces `/nvmolkit/` contents instead of creating
`/nvmolkit/<worktree-name>/`:

```bash
rsync -uavz --chmod=ugo=rwX --delete \
  --exclude="*__pycache__*" --exclude=".idea*" --exclude="*egg-info*" \
  --exclude=.git --exclude="cmake-*" --exclude="build" --exclude="build_*" \
  --exclude="_deps" --exclude="build-*" \
  "$LOCAL_SRC/" "${REMOTE_DEST}nvmolkit/"
```

`rsync` exit code 23, permission denied on delete, is expected when the
container owns some build artifacts. Tolerate it with `|| true` only when
chaining and only after confirming the failure is limited to expected delete
permissions.

### Execute A Command In The Container

```bash
ssh -J "$JUMP_HOST" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$COMPUTE_NODE" \
  'docker exec $(docker ps -q) bash -c "<COMMAND>"'
```

Replace `<COMMAND>` with the shell command. Quote carefully because the command
string passes through local ssh, remote ssh, and `docker exec`.

### Chain Sync Plus Remote Command

Prefer one shell invocation for obvious sequences such as sync, build, and run.
This keeps approvals low and keeps the remote state easy to reason about:

```bash
(rsync -uavz --chmod=ugo=rwX --delete \
  --exclude="*__pycache__*" --exclude=".idea*" --exclude="*egg-info*" \
  --exclude=.git --exclude="cmake-*" --exclude="build" --exclude="build_*" \
  --exclude="_deps" --exclude="build-*" \
  "$LOCAL_SRC" "$REMOTE_DEST<subdir>/" || true) && \
ssh -J "$JUMP_HOST" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$COMPUTE_NODE" 'docker exec $(docker ps -q) bash -c "
    cd /tmp/nvmolkit-build && \
    make -j 16 <target> 2>&1 | tail -10 && \
    echo === RUN === && \
    export NVMOLKIT_TESTDATA=/nvmolkit/tests/test_data && \
    ./tests/<binary> 2>&1 | tail -20
  "'
```

Use `&&` between every step that must succeed in order. Use `;` only when an
earlier failure is acceptable.

## Toolchain Quirks

- `cmake` is at `/usr/local/anaconda/bin/cmake`; `ninja` may not be installed,
  so fall back to Unix Makefiles and `make -j N`.
- `nvcc` is at `/usr/local/cuda/bin/nvcc`.
- `conda activate <env>` from `~/miniforge3/...` does not exist on these
  containers; the `base` environment is available.
- `nproc` typically reports a large machine, often around 256 cores. Choose
  build parallelism intentionally.

## Quote Survival

Single-quote the outer `bash -c` body once commands include many `&&` operators
or embedded `$VAR` references. Escape literal `$` as `\$` when it must survive
to the remote shell. Test once with a trivial command, such as `echo $PWD`,
before sending a long pipeline.

## Usage From Other Skills

When another skill needs remote GPU execution:

1. Read this skill first to get the connection primitives.
2. Ask the user for `COMPUTE_NODE` at the start of the session, even inside a
   multi-step optimization loop.
3. Use the standard nvMolKit mapping, `/home/scratch.kboyd_other/nvmolkit/`
   to `/nvmolkit/`, unless the user asks for a different path.
4. Use the sync plus remote exec pattern above and chain commands with `&&`.
5. Request `sandbox_permissions: "require_escalated"` on shell calls because SSH
   and rsync need network access outside the sandbox.
6. Set `timeout_ms` high enough for the expected workload. Incremental builds
   are often 30-60 seconds, full builds around 3 minutes, benchmarks 60-300
   seconds, and full `ctest` may take several minutes.

## Verifying Connectivity

Run a smoke test once after the user provides `COMPUTE_NODE`:

```bash
ssh -J "$JUMP_HOST" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$COMPUTE_NODE" 'docker exec $(docker ps -q) nvidia-smi --query-gpu=name --format=csv,noheader'
```

This should return one GPU name per visible GPU, such as `NVIDIA H100 80GB
HBM3` or `NVIDIA A100-SXM4-80GB`.
