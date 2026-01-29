for i in {0..7}; do
    session_name="gpu${i}"
    script_name="gpu${i}.sh"

    # Launch a detached tmux session that directly runs 
    # "CUDA_VISIBLE_DEVICES=$i ./$script_name" then waits in bash
    tmux new-session -d -s "$session_name" \
        "export CUDA_VISIBLE_DEVICES=$i; ./axbench/demo/lora_equal/scripts/$script_name; bash"

done

echo
echo "All 8 sessions created! Use 'tmux ls' to see them."
echo "Attach to a session with: tmux attach -t gpuN (where N is 0..7)."