#!/usr/bin/env bash

GPU_ID=$(rocm-smi --showmemuse --showuse --csv \
	     | csvcut --delete-empty-rows --columns 'GPU Memory Allocated (VRAM%),GPU use (%),device' \
	     | tail --lines=+2 \
	     | sed 's/card//' \
	     | sort --field-separator=, --key=1 --reverse \
	     | head --lines=1 \
	     | cut --delimiter=, --field=3)
echo "Less stressed GPU = ${GPU_ID}"

export ROCR_VISIBLE_DEVICES="${GPU_ID}"

echo "PyTorch GPU visibility:"
torch_list_gpus.py

export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
t="${script_dir}/test.py"

pytest --exitfirst \
    "${t}::test_op_prefill_fp8" \
    "${t}::test_op_prefill_varlen_fp8" \
    "${t}::test_op_prefill_bwd_split_impl"
