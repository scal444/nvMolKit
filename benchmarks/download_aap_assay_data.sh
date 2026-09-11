#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

destination=${1:-/data/assay_data}
archive_dir="$destination/_pubchem_archives"
novartis_source=${NOVARTIS_SOURCE:-/rdcu_profiles/aap_java_reference_2015/source_bundle/extracted/examples/NovartisMalariaBox/Novartis_GNF_NoModifier.sdf}
if [[ ! -f "$novartis_source" ]]; then
  novartis_source=/home/scratch.kboyd_other/for_download/aap_java_reference_2015/source_bundle/extracted/examples/NovartisMalariaBox/Novartis_GNF_NoModifier.sdf
fi

mkdir -p \
  "$archive_dir" \
  "$destination/novartis_malaria/raw" \
  "$destination/pubchem_aid_485297/raw" \
  "$destination/pubchem_aid_485313/raw" \
  "$destination/pubchem_aid_588342/raw" \
  "$destination/pubchem_aid_686979/raw"

cp --update=none "$novartis_source" "$destination/novartis_malaria/raw/Novartis_GNF_NoModifier.sdf"

download_archive() {
  local range=$1
  local archive="$archive_dir/$range.zip"
  echo "PROGRESS download_start archive=$range"
  curl --location --fail --retry 8 --retry-all-errors --continue-at - \
    --output "$archive" \
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/$range.zip"
  echo "PROGRESS download_finished archive=$range bytes=$(stat --format=%s "$archive")"
}

extract_aid() {
  local range=$1
  local aid=$2
  local assay_dir="$destination/pubchem_aid_$aid/raw"
  local member
  member=$(unzip -Z1 "$archive_dir/$range.zip" | awk -v aid="$aid" '$0 ~ ("(^|/)" aid "\\.csv(\\.gz)?$") { print }')
  if [[ -z "$member" ]]; then
    echo "ERROR AID $aid was not found in $range.zip" >&2
    return 1
  fi
  unzip -jo "$archive_dir/$range.zip" "$member" -d "$assay_dir"
  if [[ "$member" == *.gz ]]; then
    gzip --decompress --keep --force "$assay_dir/$aid.csv.gz"
  fi
  echo "PROGRESS extracted aid=$aid path=$assay_dir/$aid.csv"
}

download_archive 0485001_0486000
extract_aid 0485001_0486000 485297
extract_aid 0485001_0486000 485313

download_archive 0588001_0589000
extract_aid 0588001_0589000 588342

download_archive 0686001_0687000
extract_aid 0686001_0687000 686979

echo "FINISHED assay source downloads"
