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

"""
Random walk in AndroidWorld environment, collecting <screen_pre, action, screen_after> triples.
Maintain an unclickable element pool: during random walk, if the action does not change the screen, add the element to the pool.
"""

from collections.abc import Sequence
import os
import requests
import random
from typing import Type
from PIL import Image
import time
import json
import uuid
import pickle

from absl import app
from absl import flags
from absl import logging
from android_world import registry
from android_world.agents import infer
from android_world.agents import t3a
from android_world.agents import agent_utils
from android_world.agents.t3a import _generate_ui_elements_description_list_full
from android_world.env import env_launcher, json_action
from android_world.env import representation_utils
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


def generate_text_input(element_list_text, interactive_element, max_retries=3):
    # Construct the prompt with specific instructions
    def is_valid_response(response_text):
        if "\n" in response_text:
            return False
        # Optionally, add more validation rules here
        return True

    def element_to_text(element):
        """Convert the element into a text description."""
        description = ""
        if getattr(element, 'resource_id', None):
            description += f"Resource ID: {element.resource_id}\n"
        if getattr(element, 'text', None):
            description += f"Text: {element.text}\n"
        if getattr(element, 'content_description', None):
            description += f"Content Description: {element.content_description}\n"
        if getattr(element, 'class_name', None):
            description += f"Class Name: {element.class_name}\n"
        if getattr(element, 'hint_text', None):
            description += f"Hint Text: {element.hint_text}\n"
        return description or "No additional information."

    prompt = f"""
    You are an intelligent input assistant. The current UI elements are as follows:
    {element_list_text}
    The selected editable element information is as follows:
    {element_to_text(interactive_element)}
    Based on the above information, please randomly generate a text content that a user might input into this element. The text should be contextually appropriate. For example, if it's a search box, you might generate a search query; if it's a username input field, you might generate a username.
    **Please return only the generated text without any additional explanation. Do not include any prefixes or suffixes.**

    If you understand, please provide the text input now.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    payload = {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 20,
        "temperature": 0.7,
        "n": 1,
    }

    retries = 0
    while retries < max_retries:
        try:
            # Send POST request to OpenAI API
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            text_input = result['choices'][0]['message']['content'].strip()

            # Validate the response
            if is_valid_response(text_input):
                return text_input
            else:
                print(f"Invalid response format: '{text_input}'. Retrying...")
                retries += 1
                time.sleep(1)  # Wait a bit before retrying
        except Exception as e:
            print(f"Error generating text input: {e}")
            retries += 1
            time.sleep(1)

    # If all retries fail, return a default text
    print("Failed to get valid text input after retries. Returning default text.")
    return "Test Input"


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


def filter_interactive_elements(elements, screen_width_height_px, unc_elem_pool):
    interactive_elements = []
    screen_width, screen_height = screen_width_height_px

    # List of excluded package names, adding other keyboard app package names can improve filtering effect
    # Keyboard and system UI (e.g., battery indicator is actually not clickable)
    excluded_packages = {'com.google.android.inputmethod.latin', 'com.android.systemui'}

    for index, element in enumerate(elements):

        if element.package_name in excluded_packages:
            continue

        if element.is_enabled and element.is_visible:
            if element.bbox_pixels:
                x_min = element.bbox_pixels.x_min
                x_max = element.bbox_pixels.x_max
                y_min = element.bbox_pixels.y_min
                y_max = element.bbox_pixels.y_max

                # Check if bounding box is within screen bounds and coordinates are valid
                if not (x_min >= x_max or x_min >= screen_width or x_max <= 0 or
                        y_min >= y_max or y_min >= screen_height or y_max <= 0):

                    # Compute element identifier
                    element_identifier = element_to_identifier(element)
                    element_identifier_str = json.dumps(element_identifier, sort_keys=True)
                    if element_identifier_str not in unc_elem_pool:
                        interactive_elements.append([index, element])

    return interactive_elements


def sample_action_element(element, element_list_text):
    index, interactive_element = element
    actions = []
    # If element is editable, prioritize text input
    if interactive_element.is_editable:
        # Use GPT to generate appropriate text input based on current UI
        text_input = generate_text_input(element_list_text, interactive_element)
        return {"action_type": "input_text", "text": text_input, "index": index}
    # Assume all elements are clickable, add click action to action list
    actions.append({"action_type": "click", "index": index})
    # If element can be long pressed, add long press action to action list
    if interactive_element.is_long_clickable:
        actions.append({"action_type": "long_press", "index": index})
    # If multiple actions are available, choose based on given probabilities
    if actions:
        if len(actions) == 1:
            return actions[0]
        else:
            # Choose click or long press with 90% and 10% probability (assuming both actions are available)
            return random.choices(actions, weights=[9, 1], k=1)[0]


def get_task_app(task_name):
    # Choose which app to open based on task name
    task_2_app = {
        "AudioRecorder": "Audio Recorder",
        "Browser": "Files",
        "SimpleCalendar": "Simple Calendar Pro",
        "Camera": "Camera",
        "Clock": "Clock",
        "Contacts": "Contacts",
        "Expense": "Pro Expense",
        "ExpenseAddMultipleFromMarkor": "Pro Expense",
        "Files": "Files",
        "Markor": "Markor",
        "Osm": "OsmAnd",
        "Recipe": "Broccoli - Recipe App",
        "RecipeAddMultipleRecipesFromMarkor": "Broccoli - Recipe App",
        "RecipeAddMultipleRecipesFromMarkor2": "Broccoli - Recipe App",
        "Retro": "Retro Music",
        "SimpleDraw": "Simple Draw Pro",
        "SaveCopyOfReceipt": "Simple Gallery Pro",
        "SimpleSms": "Simple SMS Messenger",
        "System": "Settings",
        "Turn": "Settings",
        "Vlc": "VLC",
        "Tasks": "Tasks",
        "Notes": "Joplin",
        "Sports": "OpenTracks",
    }
    if task_name in task_2_app:
        return task_2_app[task_name]
    else:
        for task, app in task_2_app.items():
            if task in task_name:
                return app
        return "Home"


def has_screen_changed(before_elements, after_elements):
    # Convert UI elements to identifiers, excluding system UI elements
    before_set = set(
        json.dumps(element_to_identifier(elem), sort_keys=True)
        for elem in before_elements
        if elem.package_name != 'com.android.systemui'
    )
    after_set = set(
        json.dumps(element_to_identifier(elem), sort_keys=True)
        for elem in after_elements
        if elem.package_name != 'com.android.systemui'
    )
    return before_set != after_set


# Random walk pipeline:
# 1. Initialize environment
# 2. Randomly select an app to sample based on task config
# 3. Load unclickable element pool
# 4. Execute num_step steps of random walk within an app
#     a. For each step, get a set of currently executable actions based on current state
#     b. Randomly select an action
#     c. Execute action, if screen doesn't change it means it's an unclickable element, add to pool; otherwise record triples

def _main() -> None:
  """Random walk in AndroidWorld for sampling"""
  SCREEN_DIR = './screenshots_camera'
  if not os.path.exists(SCREEN_DIR):
      os.mkdir(SCREEN_DIR)

  TRAJECTORY_DIR = './trajectories_camera'
  if not os.path.exists(TRAJECTORY_DIR):
      os.mkdir(TRAJECTORY_DIR)

  PARAMS_DIR = './params_camera'
  if not os.path.exists(PARAMS_DIR):
      os.mkdir(PARAMS_DIR)

  # Launch Android emulator (ADB) and return to home screen
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
  )
  env_launcher.verify_api_level(env)
  env.reset(go_home=True)

  # Task collection
  task_registry = registry.TaskRegistry()
  aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)  # All AW tasks, total 116
  print(aw_registry)
  print("There are total {} tasks.".format(len(aw_registry)))

  aw_list = list(aw_registry.items())
  random.shuffle(aw_list)
  aw_registry = dict(aw_list)

  # Record unclickable elements for each app
  if not os.path.exists('./unclickable_elem_pool'):
      os.mkdir('./unclickable_elem_pool')

  for task_id in range(len(aw_registry)):

      task_uuid = str(uuid.uuid4())

      # Select a task
      task_name, task_type = list(aw_registry.items())[task_id]

      # Open different apps as starting state based on task
      app_name = get_task_app(task_name)

      # Initialize and return to home screen, this initialization initializes the corresponding app based on task's app snapshot
      params = task_type.generate_random_params()

      # Record random params
      with open(os.path.join(PARAMS_DIR, task_uuid+"_params.pkl"), "wb") as f:
          pickle.dump(params, f)

      task = task_type(params)
      task.initialize_task(env)
      env.reset(go_home=True)

      agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4o-2024-08-06'))  # Only used to adapt environment interface, actually no need for agent

      print("Open App: {}".format(app_name))
      print("Goal: {}".format(task.goal))
      if app_name != "Home":
          open_app_action = {"action_type": "open_app", "app_name": app_name}
          converted_action = json_action.JSONAction(**open_app_action)
          agent.env.execute_action(converted_action)
          time.sleep(3.0)

      # Load unclick_elem_pool
      unc_elem_pool_path = os.path.join('./unclickable_elem_pool', str(app_name)+".json")
      if not os.path.exists(unc_elem_pool_path):
          unc_elem_pool = set()
      else:
          with open(unc_elem_pool_path, 'r') as f:
              unc_elem_pool_list = json.load(f)
              unc_elem_pool = set(unc_elem_pool_list)

      # Random walk sampling for num_step steps
      trajectory = []
      num_step = 10
      for i in range(num_step):
          # Get current state
          env_state = agent.get_post_transition_state()
          logical_screen_size = agent.env.logical_screen_size
          ui_elements = env_state.ui_elements
          screen, element_list_text = get_state(env_state, logical_screen_size, ui_elements)
          print(element_list_text)

          # Get all executable actions on current screen: 1. Different sampling ratios & logic for different actions; 2. Keyboard elements should not be sampled, filter these elements
          # TODO: (questionable) Some samples seem to occasionally not match up?
          # Interactive elements on current screen
          interactive_elements = filter_interactive_elements(ui_elements, logical_screen_size, unc_elem_pool)
          # Other available actions on current screen
          addition_actions = [{"action_type": "scroll", "direction": "down"}, {"action_type": "scroll", "direction": "up"},
                              {"action_type": "navigate_back"}]

          # Sample an action/element
          weight_interactive = 10  # Weight for interactive elements
          weight_addition = 1  # Weight for other actions
          action_element = random.choice(interactive_elements*weight_interactive+addition_actions*weight_addition)
          if "action_type" in action_element:   # If it's a direct action
              action_sample = action_element
          else:     # If it's an element, sample an action that can be executed on it based on element properties (e.g., is_clickable)
              action_sample = sample_action_element(action_element, element_list_text)
          print("Cand Elements: {}".format([index for (index, elem) in interactive_elements]))
          print(action_sample)
          if "index" in action_sample:
              print(ui_elements[action_sample['index']])

          # Execute action
          converted_action = json_action.JSONAction(**action_sample)
          agent.env.execute_action(converted_action)    # TODO: (questionable) Some actions seem to not be fully executed? Theoretically execute_action waits for action completion
          time.sleep(2.0)

          # Check if action caused screen change
          env_state_after = agent.get_post_transition_state()
          logical_screen_size = agent.env.logical_screen_size
          ui_elements_after = env_state_after.ui_elements
          if not has_screen_changed(ui_elements, ui_elements_after):
              print("The screen not change")
              # If screen didn't change and this operation was on an element, add this element to Unavailable Ele Pool
              if "index" in action_sample:
                  element = ui_elements[action_sample['index']]
                  element_identifier = element_to_identifier(element)
                  unc_elem_pool.add(json.dumps(element_identifier, sort_keys=True))
          else:
              # Record this action
              screen_before_uuid = str(uuid.uuid4())
              screen_after_uuid = str(uuid.uuid4())
              screen_before_filename = os.path.join(SCREEN_DIR, f'{screen_before_uuid}.png')
              screen.save(screen_before_filename)

              screen_after, element_list_text_after = get_state(env_state_after, logical_screen_size, ui_elements_after)
              screen_after_filename = os.path.join(SCREEN_DIR, f'{screen_after_uuid}.png')
              screen_after.save(screen_after_filename)

              ui_elements_before_identifiers = [element_to_identifier(elem) for elem in ui_elements]
              interactive_elements_before = [element_to_identifier(elem[1]) for elem in interactive_elements]

              ui_elements_after_identifiers = [element_to_identifier(elem) for elem in ui_elements_after]
              interactive_elements_after_full = filter_interactive_elements(ui_elements_after, logical_screen_size, unc_elem_pool)
              interactive_elements_after = [element_to_identifier(elem[1]) for elem in interactive_elements_after_full]

              if "index" in action_sample:
                  action_element = element_to_identifier(ui_elements[action_sample['index']])
              else:
                  action_element = None

              step_data = {
                  'task_uuid': task_uuid,
                  'task': task_name,
                  'app': app_name,
                  'screen_before': screen_before_filename,
                  'element_list_text_before': element_list_text,
                  'ui_elements_before': ui_elements_before_identifiers,
                  'interactive_elements_before': interactive_elements_before,
                  'screen_after': screen_after_filename,
                  'element_list_text_after': element_list_text_after,
                  'ui_elements_after': ui_elements_after_identifiers,
                  'interactive_elements_after': interactive_elements_after,
                  'action': action_sample,
                  'action_element': action_element
              }

              trajectory.append(step_data)
              print(f"Recorded step {i + 1} for task {task_name}")

      # Save the updated unc_elem_pool
      with open(unc_elem_pool_path, 'w') as f:
          json.dump(list(unc_elem_pool), f, indent=2)

      # Save the trajectory to a JSON file
      trajectory_uuid = str(uuid.uuid4())
      trajectory_filename = os.path.join(TRAJECTORY_DIR, f'{task_name}_{trajectory_uuid}.json')
      with open(trajectory_filename, 'w') as f:
          json.dump(trajectory, f, indent=2)
      print(f"Saved trajectory for task {task_name} to {trajectory_filename}")

      env.reset(go_home=True)

  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
