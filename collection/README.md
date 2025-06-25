# README

In addition to the collected-synthesized data, we here provide scripts collected from the environment for Reverse Task Synthesis, helping you extend OS-Genesis to more scenarios or synthesize more data as desired.

# Mobile

## Random walk in the AndroidWorld Environment

We primarily use the AndroidWorld environment as the infrastructure for collecting triplet data.

### Step 1

First, clone the [AndroidWorld](https://github.com/google-research/android_world) repository, then place `random_walk_aw.py` and `run_random_walk_aw.py` in its directory.

It is recommended to go through the full installation process of AndroidWorld to ensure that the relevant apps are properly loaded into the virtual machine, as this directly determines the exploration space.

`random_walk_aw.py` provides our implementation logic for random walking in the environment to obtain `<screen_pre, action, screen_after>` triples. You can use `python run_random_walk_aw.py` to collect large-scale interaction triples.

### Step 2

Start the virtual machine with the following command:

```bash
EMULATOR_NAME=AndroidWorldAvd # From previous step
~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
```

Now you can see an interactable Android Virtual Machine, with multiple apps installed.

### Step 3

Then launch

```bash
python mobile_runner.py
```

After completion, you will find the following outputs in the directory, each screenshot will be saved in the (1) original format and (2) the allytree annotated format, like the example in samples folder.


## Build Trajectories in the AndroidWorld Environment

Now we have instructions, and here we can transform the instructions into trajectories.

### Step 1
To further improve the quality of data collection, we provide a pre-saved set of app states in `params_new.zip`.
Please unzip it into the android_env/android_world directory. Replace the existing folder if necessary.


### Step 2

Move your instruction file (e.g., `aw_instructions.json`) into the `android_env/androidworld` directory. This file contains the instructions generated from the random walk.

Then, launch `run_gpt_task_runner.py`. After execution, you will obtain:
1. `aw_instructions.json` — the file containing the augmented trajectory information.
2. A folder storing the corresponding screenshots, e.g., `screenshots_gpt_241103`.



> [!NOTE]  Known Issue: VM May Become Unresponsive During Long Runs
During long runs, the VM may occasionally become unresponsive, preventing further data collection. We suspect this is due to memory issues within the VM.
The easiest workaround is to reset the device directly in Android Studio to release the occupied memory and restore normal operation.
<!-- 
## Trajectory Construction

1. Install the AndroidWorld Environment as described in: https://github.com/google-research/android_world
2. Move the scripts to the AndroidWorld directory: ``android_env/android_world``
3. Run the following command to collect the data: -->
<!-- Move your instruction file, e.g., `aw_instructions.json`, to the `android_env/androidworld` directory, which contains the instructions for the random walk. -->


# Desktop

## Random walk in the WebArena Environment
First, configure [WebArena](https://github.com/web-arena-x/webarena) and open the specified ports to access the website, then place `random_walk_web.py` in its directory.

`random_walk_web.py` provides our implementation logic for random walking in the environment to obtain <screen_pre, action, screen_after> triples. You can use `python random_walk_web.py` to collect large-scale interaction triples.

# Trajectory Reward Modeling
We provide an example of the Reward Model we use in `genesis_rm.py`. For more information, please refer to the original paper.