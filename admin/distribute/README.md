# nvMolKit pip wheel build, test, and release

This directory holds the build matrix and post-build distribution tooling for
the pip wheel pipeline. The build entry point lives in `admin/deploy/`; the
RDKit build matrix, manylinux hooks, repair scripts, wheel retagging, and
simple-index generation live here.

Each release produces two artifact streams:

- **Canonical PyPI wheels**: plain `nvmolkit-<version>` wheels built against
the canonical RDKit version. These are uploaded to PyPI.
- **RDKit-pinned variant wheels**: every matrix wheel retagged with a PEP 440
local version segment like `+rdkit2025.9.6`. These are uploaded as GitHub
Release assets and exposed through a PEP 503 simple index on GitHub Pages.

PyPI rejects local-version segments, so do not upload the retagged variant
wheels to PyPI.

## Common variables

Set these once per release. Adjust `WHEELHOUSE` when using a differently named
artifact directory.

```bash
cd /home/kevin/repos/nvmolkit

VERSION=0.5.1
WHEELHOUSE="$PWD/wheelhouse_v0_5_1"
CANONICAL_RDKIT=2026.3.1
GH_REPO=NVIDIA-BioNeMo/nvMolKit
INDEX_DIR=/tmp/nvmolkit-wheels-pages-v0_5_1
BUILD_WORKTREE_ROOT=/tmp/nvmolkit_wheels_v0_5_1
PAGES_WORKTREE=/tmp/nvmolkit-pages-v0_5_1
```

## 1. Build the full wheel matrix

`build_full_matrix.sh` reads `admin/distribute/rdkit_build_matrix.yaml`, skips
RDKit `2025.3.1` through `2025.3.5`, and builds each supported `(rdkit, python)` pair in an isolated worktree with its own Conan cache. With `8 2`,
the build runs 8 wheel jobs in parallel with 2 compile threads per job, for 16
total compile threads.

```bash
conda activate nvmolkit_pip_build
export CIBW_MANYLINUX_X86_64_IMAGE=ghcr.io/nvidia-digital-bio/nvmolkit-manylinux-cuda12:2026.04.28

WHEELHOUSE="$WHEELHOUSE" \
WORKTREE_ROOT="$BUILD_WORKTREE_ROOT" \
    bash admin/deploy/build_full_matrix.sh 8 2
```

Outputs:

- wheels: `$WHEELHOUSE/rdkit<X.Y.Z>/py<M.N>/*.whl`
- build logs: `$WHEELHOUSE/logs/`
- matrix pairs and timings: `$WHEELHOUSE/jobs/`

## 2. Test the wheels

Smoke tests run against every discovered wheel. Full tests default to the
curated subset in `admin/test/full_test_subset.txt`; pass the generated matrix
pairs file to run full tests on every wheel.

```bash
conda activate nvmolkit_pip_build

bash admin/test/test_all_wheels.sh "$WHEELHOUSE" smoke
bash admin/test/test_all_wheels.sh "$WHEELHOUSE" full

# Optional: full pytest over every built matrix pair.
bash admin/test/test_all_wheels.sh "$WHEELHOUSE" full "$WHEELHOUSE/jobs/pairs.txt"
```

Outputs:

- test logs: `$WHEELHOUSE/test_logs/`
- test timings: `$WHEELHOUSE/test_logs/timings.tsv`

## 3. Stage canonical PyPI wheels

Copy the canonical RDKit wheels without retagging them. These keep the plain
project version, for example `0.5.1`.

```bash
mkdir -p "$WHEELHOUSE/pypi"
cp "$WHEELHOUSE/rdkit${CANONICAL_RDKIT}"/py*/*.whl "$WHEELHOUSE/pypi/"

twine check "$WHEELHOUSE"/pypi/*.whl
```

PyPI upload is last because filenames cannot be overwritten on PyPI.

## 4. Retag RDKit-pinned variant wheels

Retag every raw matrix wheel into `nvmolkit-<version>+rdkit<X.Y.Z>`. The retag
script rewrites `METADATA`, the `.dist-info` directory name, the auditwheel
SBOM when present, and `RECORD`.

```bash
rm -rf "$WHEELHOUSE/variants"

for variant_dir in "$WHEELHOUSE"/rdkit*/; do
    v=$(basename "$variant_dir" | sed 's/^rdkit//')
    out="$WHEELHOUSE/variants/rdkit${v}"
    mkdir -p "$out"

    for whl in "$variant_dir"/py*/*.whl; do
        python admin/distribute/retag_wheel.py "$whl" "$out" "rdkit${v}"
    done
done

find "$WHEELHOUSE/variants" -name '*.whl' | wc -l
twine check "$WHEELHOUSE"/variants/rdkit*/*.whl
```

Expected count for the current matrix is 32 variant wheels.

## 5. Generate simple-index pages

Generate one PEP 503 index per RDKit variant. The links point at the GitHub
Release assets that will be uploaded in the next step.

```bash
rm -rf "$INDEX_DIR"

RELEASE_URL="https://github.com/NVIDIA-BioNeMo/nvMolKit/releases/download/v${VERSION}"
./admin/distribute/generate_simple_index.sh \
    "$WHEELHOUSE/variants" \
    "$INDEX_DIR" \
    "$RELEASE_URL"
```

Output path shape:

```text
$INDEX_DIR/rdkit2025.9.6/simple/nvmolkit/index.html
```

## 6. Upload variant wheels to the GitHub release

Attach the retagged variant wheels to the `v${VERSION}` release on the main
repo. `--clobber` makes the upload loop rerunnable for GitHub Release assets.

```bash
gh release view "v${VERSION}" --repo "$GH_REPO"

# If the release does not exist yet, create it after the tag exists.
# gh release create "v${VERSION}" --repo "$GH_REPO" \
#     --title "nvMolKit v${VERSION}" --notes-file RELEASE_NOTES.md

find "$WHEELHOUSE/variants" -name '*.whl' -print0 |
    xargs -0 -P 8 -n 1 gh release upload "v${VERSION}" \
        --repo "$GH_REPO" --clobber
```

Verify filenames, not just asset count, before publishing the index:

```bash
find "$WHEELHOUSE/variants" -name '*.whl' -printf '%f\n' | sort \
    > /tmp/expected-nvmolkit-assets.txt

gh release view "v${VERSION}" --repo "$GH_REPO" --json assets \
    --jq '.assets[].name' | sort > /tmp/uploaded-nvmolkit-assets.txt

comm -23 /tmp/expected-nvmolkit-assets.txt /tmp/uploaded-nvmolkit-assets.txt
```

The `comm` command should print nothing.

## 7. Publish the simple-index pages

The GitHub Pages site is served from `docs/` on `github_pages_host`. The wheel
indexes live under `docs/wheels/`, producing install URLs like:

```text
https://nvidia-bionemo.github.io/nvMolKit/wheels/rdkit2025.9.6/simple/
```

Merge the generated index tree into the existing Pages wheel indexes. Do not
replace `docs/wheels/`: each `simple/nvmolkit/index.html` page must retain
links for older nvMolKit releases as well as the new release.

```bash
REMOTE=origin

git fetch "$REMOTE" github_pages_host
git worktree add -B "publish-v${VERSION}-wheels" "$PAGES_WORKTREE" \
    "$REMOTE/github_pages_host"

python admin/distribute/merge_simple_index.py \
    "$INDEX_DIR" \
    "$PAGES_WORKTREE/docs/wheels"

git -C "$PAGES_WORKTREE" add docs/wheels
git -C "$PAGES_WORKTREE" diff --staged --stat
git -C "$PAGES_WORKTREE" commit -m "Publish v${VERSION} variant wheel indexes"
git -C "$PAGES_WORKTREE" push "$REMOTE" HEAD:github_pages_host
git worktree remove "$PAGES_WORKTREE"
```

The staged diff should add or update links for `v${VERSION}` while keeping
older release links in the same PEP 503 project pages. Re-run index generation
and Pages publication whenever variant wheels are re-uploaded, because the
index includes wheel hashes. The merge helper replaces links with the same wheel
filename, so rerunning after rebuilding a wheel updates its hash without
dropping links for other versions.

When rebuilding Sphinx docs on `github_pages_host`, preserve `docs/wheels/`:

```bash
rsync -a --delete --exclude='wheels/' <sphinx-build>/ docs/
```

## 8. Upload canonical wheels to PyPI

Upload only the plain canonical wheels in `$WHEELHOUSE/pypi`. Do not upload the
retagged variant wheels.

```bash
twine check "$WHEELHOUSE"/pypi/*.whl

# Optional dry run against TestPyPI.
twine upload --repository testpypi "$WHEELHOUSE"/pypi/*.whl

# Real PyPI upload.
twine upload "$WHEELHOUSE"/pypi/*.whl
```

PyPI does not allow re-uploading the same filename, even after deletion. If a
canonical wheel has already been uploaded and needs to change, bump the project
version first.

## Script reference

Build pipeline:

- `admin/deploy/build_full_matrix.sh`: top-level parallel matrix driver.
- `admin/deploy/build_one_wheel.sh`: per-pair worker used by the matrix driver.
- `admin/deploy/build_pip_wheels.sh`: cibuildwheel driver for one RDKit
version.

Build hooks and metadata:

- `admin/distribute/rdkit_build_matrix.yaml`: supported RDKit/Python matrix.
- `admin/distribute/cibuildwheel_before_build.sh`: manylinux before-build hook.
- `admin/distribute/lookup_rdkit_pypi_tag.py`: maps matrix entries to
`rdkit-pypi` tags.
- `admin/distribute/repair_wheel.sh`: auditwheel repair and repack step.

Post-build distribution:

- `admin/distribute/retag_wheel.py`: adds the `+rdkit<X.Y.Z>` local version
segment to a wheel.
- `admin/distribute/generate_simple_index.sh`: emits PEP 503 simple-index
pages from retagged variant wheels.

Post-build testing:

- `admin/test/test_all_wheels.sh`: runs smoke or full tests across wheel pairs.
- `admin/test/test_one_wheel.sh`: installs and tests one wheel pair.
- `admin/test/smoke_check.py`: minimal import and CUDA smoke probe.
- `admin/test/full_test_subset.txt`: curated full-test subset.
