#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

pytest "${script_dir}/test.py::test_op_prefill_fwd_impl_fp8" | \
awk '
BEGIN {
    has_failing_test = has_mismatched_elems = has_abs_diff = has_rel_dif = 0
    FS = "_ |[\\[\\-\\]]| _| +| / | +\\(|%\\)"
    OFS = ","
    print "Z", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD", "causal", "dropout_p", "layout", "use_exp2", "scale_per_head",
          "mismatched_elems", "total_elems", "mismatched_percentage",
          "greatest_abs_diff", "greatest_rel_diff"
}

# Match test that is failing.
/^_ test_op_prefill_fwd_impl_fp8\[(.*)\] _$/ {
    DEBUG_INPUT = $3
    output_dtype = $4
    input_dtype = $5
    scale_per_head = $6
    use_exp2 = $7
    layout = $8
    dropout_p = $9
    causal = $10
    Z = $11
    HQ = $12
    HK = $13
    N_CTX_Q = $14
    N_CTX_K = $15
    D_HEAD = $16
    has_failing_test = 1
    has_mismatched_elems = has_abs_diff = has_rel_dif = 0
}

# Match mismatched elements.
/^E +Mismatched elements: [0-9]+ \/ [0-9]+ \([0-9]+\.[0-9]+%\)$/ {
    if (has_failing_test) {
        mismatched_elems = $4
        total_elems = $5
        mismatched_percentage = $6
        has_mismatched_elems = 1
        has_abs_diff = has_rel_dif = 0
    }
}

# Match greatest absolute difference.
/^E +Greatest absolute difference: [0-9]+\.[0-9].+at index.+$/ {
    if (has_mismatched_elems) {
        abs_diff = $5
        has_abs_diff = 1
        has_rel_dif = 0
    }
}

# Match greatest relative difference.
/^E +Greatest relative difference: [0-9]+\.[0-9].+at index.+$/ {
    if (has_abs_diff) {
        rel_diff = $5
        has_rel_dif = 1
    }
}

# Generate output after all patterns are matched.
{
    if (has_failing_test && has_mismatched_elems && has_abs_diff && has_rel_dif) {
        print Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, layout, use_exp2, scale_per_head,
              mismatched_elems, total_elems, mismatched_percentage,
              abs_diff, rel_diff
        has_failing_test = has_mismatched_elems = has_abs_diff = has_rel_dif = 0
    }
}
'
