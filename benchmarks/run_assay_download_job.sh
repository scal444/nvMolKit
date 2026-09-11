#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

destination=${1:-/data/assay_data}
log_path="$destination/download.log"
metadata_path="$destination/download.meta"
mkdir -p "$destination"
exec >>"$log_path" 2>&1

echo "START time=$(date --iso-8601=seconds) pid=$$"
printf 'pid=%s\nstart_time=%s\ncommand=%q\ndestination=%s\nlog=%s\n' \
  "$$" \
  "$(date --iso-8601=seconds)" \
  "bash benchmarks/download_aap_assay_data.sh $destination" \
  "$destination" \
  "$log_path" >"$metadata_path"

bash benchmarks/download_aap_assay_data.sh "$destination"
exit_code=$?
echo "FINISHED exit_code=$exit_code time=$(date --iso-8601=seconds)"
exit "$exit_code"
