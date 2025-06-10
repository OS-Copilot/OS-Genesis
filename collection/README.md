# README

In addition to the collected-synthesized data, we here provide scripts collected from the environment for Reverse Task Synthesis, helping you extend OS-Genesis to more scenarios or synthesize more data as desired.

# Mobile

## Random walk in the AndroidWorld Environment
First, clone the [AndroidWorld](https://github.com/google-research/android_world) repository, then place `random_walk_aw.py` and `run_random_walk_aw.py` in its directory.

`random_walk_aw.py` provides our implementation logic for random walking in the environment to obtain `<screen_pre, action, screen_after>` triples. You can use `python run_random_walk_aw.py` to collect large-scale interaction triples.
## Reverse Task Synthesis

Work in progress.

## Trajectory Construction

1. Install the AndroidWorld Environment as described in: https://github.com/google-research/android_world
2. Move the scripts to the AndroidWorld directory: ``android_env/android_world``
3. Run the following command to collect the data:
```bash
python mobile_runner.py
```

# Desktop

## Random walk in the WebArena Environment
First, configure [WebArena](https://github.com/web-arena-x/webarena) and open the specified ports to access the website, then place `random_walk_web.py` in its directory.

`random_walk_web.py` provides our implementation logic for random walking in the environment to obtain <screen_pre, action, screen_after> triples. You can use `python random_walk_web.py` to collect large-scale interaction triples.

# Reward Model
We provide an example of the Reward Model we use in `genesis_rm.py`. For more information, please refer to the original paper.