# Agent instructions

## Always give the commands

Whenever you say you will do something (run a training job, evaluate weights, add a feature, run tests, and so on), also give the user the exact commands to do it. Put them in a fenced `bash` block, in the order they should run, with the `conda activate minesweeper-rl` step first when Python is involved. Give the commands even when you run them yourself, so the user can reproduce or re-run them without asking.
