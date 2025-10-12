#!/bin/bash
# GCC wrapper to add CUDA library paths for Triton compilation

# Reconstruct the command with library paths in the right place
# We need to add -L flags before -l flags for linking to work
args=()
lib_paths_added=0
for arg in "$@"; do
    if [[ "$arg" == -l* ]] && [[ $lib_paths_added -eq 0 ]]; then
        # Before any -l flag, add our library paths
        # Use both the real library and stub library paths
        args+=("-L/usr/lib/x86_64-linux-gnu" "-L/lib/x86_64-linux-gnu" "-L/usr/local/cuda/targets/x86_64-linux/lib/stubs")
        lib_paths_added=1
    fi
    args+=("$arg")
done

# Execute gcc with modified arguments
exec /usr/bin/gcc "${args[@]}"
