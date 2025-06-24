# AndroidWorld Evaluation

Unlike AndroidControl, which is a static benchmark, evaluating AndroidWorld requires aligning the action space of the trained model with the environment using `infer.py`. The easiest way is to replace the original `infer.py` under the path `android_env/android_world/android_world/agents with your customized infer.py`, and then directly run inference using original m3a.

https://github.com/google-research/android_world