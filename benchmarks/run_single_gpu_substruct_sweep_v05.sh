#!/usr/bin/env bash
set -uo pipefail

CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Set CONDA_SH to the path of conda.sh" >&2
    exit 1
fi

source "$CONDA_SH"
conda activate "${CONDA_ENV:-rdcu_dev}"

NVMOLKIT_ROOT="${NVMOLKIT_ROOT:-/nvmolkit}"
SMARTS_DIR="${SMARTS_DIR:-$NVMOLKIT_ROOT/tests/test_data/SMARTS}"
SMILES="${SMILES:-/data/enamine_real_10M.cxsmiles}"
OUT_DIR="${OUT_DIR:-/rdcu_profiles/substruct_2026_08_16_v05}"

mkdir -p "$OUT_DIR/results" "$OUT_DIR/logs"
cd "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=14
export OMP_DYNAMIC=FALSE

# Avoid nested math-library pools consuming cores beyond the explicitly
# allocated substructure worker and preprocessing pools.
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -f "$SMILES" ]]; then
    echo "Missing SMILES input: $SMILES" >&2
    exit 1
fi

declare -a DATASETS=(
    bms
    fragment_descriptors
    tautomer_transforms
    torsion_preferences
    pains
)

declare -A SMARTS_FILES=(
    [bms]="BMS_2006_filter_supported.txt"
    [fragment_descriptors]="rdkit_fragment_descriptors_supported.txt"
    [tautomer_transforms]="rdkit_tautomer_transforms_supported.txt"
    [torsion_preferences]="rdkit_torsionPreferences_v2_supported.txt"
    [pains]="wehi_pains_supported.txt"
)

declare -A MODES=(
    [bms]="hasSubstructMatch"
    [fragment_descriptors]="countSubstructMatches"
    [tautomer_transforms]="getSubstructMatches"
    [torsion_preferences]="getSubstructMatches"
    [pains]="hasSubstructMatch"
)

declare -A NUM_MOLS=(
    [bms]=860160
    [fragment_descriptors]=1548288
    [tautomer_transforms]=393216
    [torsion_preferences]=65536
    [pains]=401408
)

BATCH_SIZES=(2048 4096 8192)
WORKER_COUNTS=(4 6 8 10)

SUMMARY="$OUT_DIR/summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
    printf 'benchmark\tstatus\texit_code\tduration_s\tresult\tlog\n' > "$SUMMARY"
fi

failures=0

for dataset in "${DATASETS[@]}"; do
    smarts="$SMARTS_DIR/${SMARTS_FILES[$dataset]}"
    mode="${MODES[$dataset]}"
    num_mols="${NUM_MOLS[$dataset]}"

    if [[ ! -f "$smarts" ]]; then
        echo "Missing SMARTS input: $smarts" >&2
        exit 1
    fi

    for batch_size in "${BATCH_SIZES[@]}"; do
        for workers in "${WORKER_COUNTS[@]}"; do
            prep_threads=$((14 - workers))
            name="${dataset}_v05_b${batch_size}_w${workers}_p${prep_threads}"
            result="$OUT_DIR/results/${name}.csv"
            log="$OUT_DIR/logs/${name}.log"

            if [[ -s "$result" ]]; then
                echo "[$name] skipped: result already exists"
                continue
            fi

            echo "[$name] starting"
            echo "  molecules=$num_mols mode=$mode batch=$batch_size workers=$workers prep=$prep_threads"

            start_s=$(date +%s)

            if python "$NVMOLKIT_ROOT/benchmarks/substruct_bench.py" \
                --smiles "$SMILES" \
                --smarts "$smarts" \
                --num_mols "$num_mols" \
                --seed 42 \
                --sanitize \
                --mode "$mode" \
                --batch_size "$batch_size" \
                --workers "$workers" \
                --prep_threads "$prep_threads" \
                --num_gpus 1 \
                --runs 3 \
                --warmup \
                --no_rdkit \
                --no_validate \
                > "$log" 2>&1
            then
                # v0.5 has no --output option. Its final stdout section is a
                # CSV header followed by the result row, so recover that block
                # from the complete log into the per-configuration CSV file.
                sed -n '/method,mode,smarts,input_file/,$p' "$log" > "$result"
                if [[ -s "$result" ]]; then
                    status=ok
                    code=0
                else
                    status=fail
                    code=1
                    failures=$((failures + 1))
                    echo "[$name] benchmark completed but no CSV block was found" >&2
                    tail -n 20 "$log" >&2
                fi
            else
                status=fail
                code=$?
                failures=$((failures + 1))
                tail -n 20 "$log" >&2
            fi

            end_s=$(date +%s)
            duration=$((end_s - start_s))

            printf '%s\t%s\t%d\t%d\t%s\t%s\n' \
                "$name" "$status" "$code" "$duration" "$result" "$log" \
                >> "$SUMMARY"

            echo "[$name] $status (${duration}s)"
        done
    done
done

echo
echo "Sweep complete: $OUT_DIR"
echo "Failures: $failures"
exit "$failures"
