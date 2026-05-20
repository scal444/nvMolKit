# nvMolKit pip wheel build & release

This directory holds the build matrix and post-build tooling for the pip wheel
pipeline. The entry-point build script lives one level up at
`admin/deploy/build_pip_wheels.sh`; everything else below it (auditwheel
repair, retagging into RDKit variants, PEP 503 index generation) lives here.

End-to-end, each release produces two distinct artifact streams:

1. **Canonical PyPI release** — a single `nvmolkit-<version>` ABI built against
   one canonical RDKit version (currently 2026.3.1). Plain version, no local
   segment. Uploaded to <https://pypi.org/project/nvmolkit/>.
2. **Variant index** — every wheel in `rdkit_build_matrix.yaml` rebuilt against
   its target RDKit version, retagged with a PEP 440 local segment
   (`+rdkit<X.Y.Z>`), and exposed via a PEP 503 simple index. The wheel files
   themselves live as GitHub Release assets on the main
   [`nvmolkit`](https://github.com/NVIDIA-Digital-Bio/nvmolkit) repo's per-version
   release (e.g. `v0.5.0`); the index pages are served from the same repo's
   GitHub Pages site under `wheels/`, alongside the Sphinx docs tree on the
   `github_pages_host` branch.

PyPI rejects local-version segments, so the same wheel cannot satisfy both
streams. The canonical PyPI wheel is the unmodified build output; the variant
wheels are post-processed by `retag_wheel.py`.

## 1. Build all variants

`admin/deploy/build_full_matrix.sh` fans the full `(rdkit, python)` matrix out
across N parallel cibuildwheel workers, each in its own throwaway copy of the
source tree with its own conan2 cache. Per-pair logs and a per-pair timings TSV land
under `wheelhouse/{logs,jobs}/`. Wheels land under
`wheelhouse/rdkit<X.Y.Z>/py<M.N>/`.

```bash
conda activate nvmolkit_pip_build   # must provide cibuildwheel + wheel
./admin/deploy/build_full_matrix.sh 8 2   # 8 parallel jobs, 2 threads each
```

`build_pip_wheels.sh` (invoked per pair by `build_one_wheel.sh`) temporarily
injects `rdkit==${RDKIT_VERSION}` into `pyproject.toml`'s `Requires-Dist` for
the duration of the cibuildwheel run, so each output wheel ships with the
correct RDKit pin baked into its metadata.

The matrix YAML lists RDKit versions back to 2025.3.1, but
`build_full_matrix.sh` skips 2025.3.1 through 2025.3.5: those rdkit-pypi tags
use conan-1 invocation syntax which the manylinux+CUDA image's conan-2 cannot
parse. First buildable tag is 2025.3.6.

## 2. Smoke-test every wheel

`admin/test/test_all_wheels.sh` installs each wheel into a throwaway pip venv
built on top of a per-python-version conda interpreter env (auto-created if
missing) and runs the smoke check. Full pytest can be opted into via the
`full` or `both` modes; the full sweep set is in
`admin/test/full_test_subset.txt`.

```bash
./admin/test/test_all_wheels.sh wheelhouse smoke
# Optional: full pytest on the curated (rdkit, py) subset
./admin/test/test_all_wheels.sh wheelhouse full
```

Per-pair logs and a timings TSV land in `wheelhouse/test_logs/`.

## 3. Stage the canonical PyPI wheel set

Pick the canonical RDKit version (currently 2026.3.1) and copy its wheels aside
without retagging:

```bash
CANONICAL=2026.3.1
mkdir -p wheelhouse/pypi
cp wheelhouse/rdkit${CANONICAL}/py*/*.whl wheelhouse/pypi/
```

These wheels keep their plain `0.5.0` version and are PyPI-uploadable as-is.

## 4. Retag every variant

For each `(rdkit, py)` pair, rewrite the wheel's Version field, dist-info
directory, auditwheel SBOM, and RECORD to add the `+rdkit<X.Y.Z>` local
segment. Output goes to `wheelhouse/variants/rdkit<X.Y.Z>/`:

```bash
for variant_dir in wheelhouse/rdkit*/; do
    v=$(basename "${variant_dir}" | sed 's/^rdkit//')
    out="wheelhouse/variants/rdkit${v}"
    mkdir -p "${out}"
    for whl in "${variant_dir}"py*/*.whl; do
        python admin/distribute/retag_wheel.py "${whl}" "${out}" "rdkit${v}"
    done
done
```

The retag script recomputes RECORD from scratch, so any stale RECORD entries
from the auditwheel/patchelf step in `repair_wheel.sh` are silently corrected
here.

## 5. Generate the simple-repository indexes

Emit one `simple/nvmolkit/index.html` per variant, hashing the wheels and
pointing each `<a href>` at the GitHub Release asset URL the wheels will live
at after step 6.

```bash
VERSION=0.5.0
RELEASE_URL=https://github.com/NVIDIA-Digital-Bio/nvmolkit/releases/download/v${VERSION}
./admin/distribute/generate_simple_index.sh \
    wheelhouse/variants \
    /tmp/nvmolkit-wheels-pages \
    "${RELEASE_URL}"
```

## 6. Upload variant wheels to the main repo's GitHub Release

Attach every retagged wheel as an asset on the existing `v${VERSION}` release
on `NVIDIA-Digital-Bio/nvmolkit`. PEP 440 `+` characters survive in filenames;
`gh` URL-encodes them. `--clobber` lets the upload loop be re-run safely if a
name collides.

```bash
VERSION=0.5.0
find wheelhouse/variants -name '*.whl' -print0 |
    xargs -0 -P 8 -n 1 gh release upload "v${VERSION}" \
        --repo NVIDIA-Digital-Bio/nvmolkit --clobber
```

If the release does not yet exist, create it with `gh release create
v${VERSION} --repo NVIDIA-Digital-Bio/nvmolkit ...` first. Uploading assets to
an existing non-draft release does not change its tag, title, or notes.

Verify the upload count matches the file count before publishing the index:

```bash
expected=$(find wheelhouse/variants -name '*.whl' | wc -l)
uploaded=$(gh release view "v${VERSION}" --repo NVIDIA-Digital-Bio/nvmolkit --json assets --jq '.assets | length')
test "${expected}" = "${uploaded}"
```

Unlike PyPI, GitHub Release assets can be replaced under the same filename,
so the project version does not need to be bumped to re-cut a release. The
asset-count check above compares totals only and will overcount if any
non-variant assets are also attached to `v${VERSION}` (source tarballs, etc.);
if the matrix has shrunk since the previous upload, stale assets from prior
runs can also mask a missing current wheel. Compare filename sets instead
when re-uploading.

## 7. Publish the simple-repository indexes

The main `nvmolkit` repo's GitHub Pages site is served from `docs/` on the
`github_pages_host` branch — the same tree that hosts the Sphinx docs. The
wheel indexes live in a sibling subdirectory, `docs/wheels/`, so the
per-variant URL is
`https://nvidia-digital-bio.github.io/nvmolkit/wheels/rdkit<X.Y.Z>/simple/`.
The existing `docs/.nojekyll` at the Pages root covers `docs/wheels/`, so
the `simple/` subdirectories aren't filtered out.

Replace the wheel-index tree wholesale on `github_pages_host`:

```bash
git fetch origin github_pages_host
git worktree add /tmp/nvmolkit-pages github_pages_host
rm -rf /tmp/nvmolkit-pages/docs/wheels
mkdir -p /tmp/nvmolkit-pages/docs/wheels
cp -r /tmp/nvmolkit-wheels-pages/. /tmp/nvmolkit-pages/docs/wheels/
git -C /tmp/nvmolkit-pages add docs/wheels
git -C /tmp/nvmolkit-pages commit -m "Publish v${VERSION} variant wheel indexes"
git -C /tmp/nvmolkit-pages push origin github_pages_host
git worktree remove /tmp/nvmolkit-pages
```

Re-running steps 5–7 against the same `v${VERSION}` is safe: the hashes in
`index.html` change whenever wheels in step 6 are re-uploaded, so steps 6 and
7 should be re-run as a pair.

**Docs refresh interaction.** Sphinx doc updates on `github_pages_host`
replace the contents of `docs/` with a fresh build. The refresh must preserve
`docs/wheels/`, e.g. with

```bash
rsync -a --delete --exclude='wheels/' <sphinx-build>/ docs/
```

otherwise the next docs commit will silently drop the wheel indexes.

## 8. Upload the canonical wheel set to PyPI

Smoke-test against TestPyPI first if the build process has changed:

```bash
twine check wheelhouse/pypi/*.whl
twine upload --repository testpypi wheelhouse/pypi/*.whl
```

Then upload to public PyPI:

```bash
twine upload wheelhouse/pypi/*.whl
```

PyPI does not allow re-uploading the same filename, even after deletion. If a
canonical upload needs to change, bump the project version in
`pyproject.toml` first.

## Script reference

Build pipeline (`admin/deploy/`):

- `build_full_matrix.sh` — top-level parallel driver. Reads the matrix YAML,
  fans `(rdkit, py)` pairs out to `build_one_wheel.sh` via `xargs -P`, writes
  per-pair logs + timings TSV under `wheelhouse/{logs,jobs}/`, and prints a
  summary at the end.
- `build_one_wheel.sh` — per-pair worker. Rsyncs the repo into a throwaway
  source-tree copy, drops the working-tree `pyproject.toml` into it, sets up
  an isolated conan2 cache, and calls `build_pip_wheels.sh` for one
  `(rdkit, py)` pair.
- `build_pip_wheels.sh` — cibuildwheel driver for one RDKit version. Injects
  the matching `rdkit==<X.Y.Z>` pin into `pyproject.toml` for the duration of
  the run.

Build hooks (`admin/distribute/`):

- `cibuildwheel_before_build.sh` — runs inside the manylinux container to
  reproduce the rdkit-pypi build at the tag from `rdkit_build_matrix.yaml`.
- `repair_wheel.sh` — auditwheel + patchelf, repacked via `wheel pack` so
  RECORD reflects the post-patchelf file contents.
- `lookup_rdkit_pypi_tag.py` — helper used by the before-build hook to map an
  `rdkit_build_matrix.yaml` entry to a `kuelumbus/rdkit-pypi` git tag.
- `rdkit_build_matrix.yaml` — single source of truth for which `(rdkit, py)`
  combinations are supported.

Post-build distribution (`admin/distribute/`):

- `retag_wheel.py` — adds a PEP 440 local segment to a wheel, recomputing
  RECORD.
- `generate_simple_index.sh` — emits PEP 503 simple-repository pages from a
  tree of `rdkit<X.Y.Z>/*.whl` directories.

Post-build testing (`admin/test/`):

- `test_all_wheels.sh` — drives `test_one_wheel.sh` over a list of pairs,
  auto-creating per-python conda interpreter envs as needed.
- `test_one_wheel.sh` — pip-installs one wheel into a throwaway venv on top
  of the matching conda interpreter and runs smoke or full pytest.
- `smoke_check.py` — minimal import + GPU-op probe; passes if nvmolkit loads
  and a small CUDA operation succeeds.
- `full_test_subset.txt` — curated `(rdkit, py)` list for the full pytest
  sweep (one pair per python version, spanning the rdkit range).
