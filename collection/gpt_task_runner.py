# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a single task.

The minimal_run.py module is used to run a single task, it is a minimal version
of the run.py module. A task can be specified, otherwise a random task is
selected.
"""

from collections.abc import Sequence
import os
import random
from typing import Type
import json
import time
import uuid
from PIL import Image
import numpy as np
import re
import pickle

from absl import app
from absl import flags
from absl import logging
from android_world import registry
from android_world.agents import infer
from android_world.agents import t3a, m3a
# from android_world.agents import m3a_origin
from android_world.agents import m3a_utils
from android_world.agents.t3a import _generate_ui_elements_description_list_full
from android_world.env import env_launcher, json_action
from android_world.task_evals import task_eval

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


def _find_adb_directory() -> str:
  """Returns the directory where adb is located."""
  potential_paths = [
      os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
      os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
  ]
  for path in potential_paths:
    if os.path.isfile(path):
      return path
  raise EnvironmentError(
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
  )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)

_TASK = flags.DEFINE_string(
    'task',
    None,
    'A specific task to run.',
)


def save_image(image, directory):
    """保存图像并返回文件名"""
    unique_id = str(uuid.uuid4())
    image_name = f"{unique_id}.png"
    image_path = os.path.join(directory, image_name)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    image.save(image_path)
    return image_name


def get_state(env_state, logical_screen_size, ui_elements):
    element_list_text = _generate_ui_elements_description_list_full(
        ui_elements,
        logical_screen_size,
    )
    screen = env_state.pixels.copy()
    screen = Image.fromarray(screen.astype('uint8'))
    return screen, element_list_text


def element_to_identifier(element):
    """Converts an element to a JSON-serializable identifier."""
    bbox = getattr(element, 'bbox_pixels', None)
    bbox_dict = {'x_min': bbox.x_min, 'x_max': bbox.x_max, 'y_min': bbox.y_min, 'y_max': bbox.y_max} if bbox else None
    identifier = {
        'resource_id': getattr(element, 'resource_id', None),
        'text': getattr(element, 'text', None),
        'content_description': getattr(element, 'content_description', None),
        'class_name': getattr(element, 'class_name', None),
        'bbox_pixels': bbox_dict,
        'hint_text': getattr(element, 'hint_text', None),
        'is_checkable': getattr(element, 'is_checkable', None),
        'is_enabled': getattr(element, 'is_enabled', None),
        'is_visible': getattr(element, 'is_visible', None),
        'is_clickable': getattr(element, 'is_clickable', None),
        'is_editable': getattr(element, 'is_editable', None),
        'is_focused': getattr(element, 'is_focused', None),
        'is_focusable': getattr(element, 'is_focusable', None),
        'is_long_clickable': getattr(element, 'is_long_clickable', None),
        'is_scrollable': getattr(element, 'is_scrollable', None),
        'is_selected': getattr(element, 'is_selected', None),
        'package_name': getattr(element, 'package_name', None),
        'resource_name': getattr(element, 'resource_name', None),
    }
    return identifier


def _main() -> None:

  instruction_path = './aw_instructions_v1_sqs_p2.json'
  aw_instrcutions = json.load(open(instruction_path, 'r'))

  SCREEN_GPT_DIR = './screenshots_gpt_v1_p2'
  if not os.path.exists(SCREEN_GPT_DIR):
      os.mkdir(SCREEN_GPT_DIR)

  """Initialize Env."""
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )
  env_launcher.verify_api_level(env)

  for task_item in aw_instrcutions:

      total_tasks = len(aw_instrcutions)
      annotated_tasks = len([item for item in aw_instrcutions if "gpt_traj" in item])
      print(f"Total task: {total_tasks} --- Annotated task: {annotated_tasks}")
      if "gpt_traj" in task_item or "task_fail" in task_item:
          continue

      try:
          env.reset(go_home=True)
          task_registry = registry.TaskRegistry()
          aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

          # Initialize and Launch Apps Based on the Sampled Instructions
          app_name = task_item["app_name"]
          task_name = task_item["task_name"]
          instrcution = task_item["refine_task"]

          if task_name and task_name != "default":
            if task_name not in aw_registry:
              raise ValueError('Task {} not found in registry.'.format(_TASK.value))
            task_type: Type[task_eval.TaskEval] = aw_registry[task_name]
          else:
            task_type: Type[task_eval.TaskEval] = random.choice(
                list(aw_registry.values())
            )
            print("unknown task name")
            input()
          print(task_type)

          # load params
          task_id = task_item["task_id"]
          params_dir = './params_new'
          params_path = os.path.join(params_dir, task_id + "_params.pkl")
          with open(params_path, 'rb') as f:
              params = pickle.load(f)
          print(params)
          #params = task_type.generate_random_params()

          task = task_type(params)

          task.initialize_task(env)
          agent = m3a.M3A(env, infer.Gpt4Wrapper('gpt-4o-2024-08-06'))

          # Open the corresponding app after initializing the task.
          open_app = True
          if open_app:
              open_app_action = {"action_type": "open_app", "app_name": app_name}
              converted_action = json_action.JSONAction(**open_app_action)
              agent.env.execute_action(converted_action)
              time.sleep(3.0)

          print('Goal: ' + str(instrcution))
          is_done = False
          gpt_traj = []
          for i, _ in enumerate(range(15)):

            # Fetch the environment state before execution to stay synchronized with our training setup.
            env_state = agent.get_post_transition_state()
            logical_screen_size = agent.env.logical_screen_size
            ui_elements = env_state.ui_elements
            screen, element_list_text = get_state(env_state, logical_screen_size, ui_elements)
            screen_before = save_image(screen, SCREEN_GPT_DIR)

            # Note: The state representation here follows the M3A implementation, keeping it consistent with the agent’s observation, to ensure that the model-generated actions can correctly locate the corresponding elements.
            ui_elements_before_identifiers = [element_to_identifier(elem) for elem in ui_elements if m3a_utils.validate_ui_element(elem, logical_screen_size)]

            # take one steo
            response = agent.step(instrcution)

            # Extract screen, text, and generated action from the response.
            screen_before_som = save_image(response.data["before_screenshot_with_som"], SCREEN_GPT_DIR)
            action_prompt = response.data["action_prompt"]
            action_output = response.data["action_output"]
            action_reason = response.data["action_reason"]
            summary_prompt = response.data["summary_prompt"]
            summary = response.data["summary"]

            match = re.search(r'Action:\s*(\{.*\})', action_output)
            action_json = match.group(1) if match else "action_not_match"
            
            # Terminate if the same action is repeated three times consecutively.
            if i >= 2 and (action_json == gpt_traj[i-1]["action_json"] == gpt_traj[i-2]["action_json"]):
                break

            step_data = {
                "screen_before": screen_before,
                "screen_before_som": screen_before_som,
                "ui_elements_before_text": element_list_text,
                "ui_elements_before": ui_elements_before_identifiers,
                "action_prompt": action_prompt,
                "action_output": action_output,
                "action_json": action_json,
                "action_reason": action_reason,
                "summary_prompt": summary_prompt,
                "summary": summary
            }
            gpt_traj.append(step_data)

            if response.done:
              is_done = True
              break

          """
          agent_successful = is_done and task.is_successful(env) == 1
          print(
              f'{"Task Successful ✅" if agent_successful else "Task Failed ❌"};'
              f' {task.goal}'
          )
          """

          # env.close()

          # After each trajectory is processed, the GPT-generated annotations will be updated and written back into the original aw_instructions.json file.
          task_item["gpt_traj"] = gpt_traj
          json.dump(aw_instrcutions, open(instruction_path, 'w'))

      except Exception as e:
          print(f"An error occurred: {e}")
          task_item["task_fail"] = "fail"
          json.dump(aw_instrcutions, open(instruction_path, 'w'))
          time.sleep(10)
          break     # relaunch

def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)