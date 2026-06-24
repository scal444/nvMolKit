#!/bin/bash
# Generate PEP 503 simple-repository index pages for each RDKit variant.
#
# For every subdirectory of <variants_root> matching `rdkit<X.Y.Z>`, emits
# <out_dir>/rdkit<X.Y.Z>/simple/nvmolkit/index.html linking to the variant's
# wheels via <release_url_base>/<filename>#sha256=<hash>.
#
# Usage:
#   generate_simple_index.sh <variants_root> <out_dir> <release_url_base>
#
# Example:
#   generate_simple_index.sh \
#       wheelhouse/variants \
#       /tmp/nvmolkit-wheels-pages \
#       https://github.com/NVIDIA-Digital-Bio/nvmolkit/releases/download/v0.5.0

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <variants_root> <out_dir> <release_url_base>" >&2
    exit 1
fi

VARIANTS_ROOT="$1"
OUT_DIR="$2"
RELEASE_URL_BASE="$3"

count=0
for variant_dir in "${VARIANTS_ROOT}"/rdkit*/; do
    [ -d "${variant_dir}" ] || continue
    variant_name=$(basename "${variant_dir}")
    rdkit_ver="${variant_name#rdkit}"

    out="${OUT_DIR}/rdkit${rdkit_ver}/simple/nvmolkit"
    mkdir -p "${out}"

    {
        echo '<!DOCTYPE html>'
        echo '<html><head><meta name="pypi:repository-version" content="1.0"></head><body>'
        for whl in "${variant_dir}"*.whl; do
            [ -f "${whl}" ] || continue
            fname=$(basename "${whl}")
            hash=$(sha256sum "${whl}" | cut -d' ' -f1)
            # PEP 440 local versions contain '+', which must be percent-encoded in URLs.
            url_fname="${fname//+/%2B}"
            echo "<a href=\"${RELEASE_URL_BASE}/${url_fname}#sha256=${hash}\">${fname}</a><br>"
        done
        echo '</body></html>'
    } > "${out}/index.html"

    count=$((count + 1))
done

echo "Generated ${count} simple-index page(s) under ${OUT_DIR}"
