# AndroidWorld Evaluation

Unlike AndroidControl, which is a static benchmark, evaluating AndroidWorld requires aligning the action space of the trained model with the environment. The easiest way is 


1. Move `genesis_agent.py` and `genesis_utils.py` to the path `android_env/android_world/android_world/agents`, which is the original path of the agent in AndroidWorld.
2. Replace the original `infer.py` under the path `android_env/android_world/android_world/agents`, and then directly run inference using original m3a.

and use the `run.py` to run the evaluation:

```
python run.py \
  --suite_family=android_world \
  --agent_name=genesis_agent \
  --perform_emulator_setup \
```


We mainly follow: https://github.com/google-research/android_world



