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

"""Utils for Genesis."""

import base64
import re
from typing import Any, Optional
from android_world.env import representation_utils
import cv2
import numpy as np


def _logical_to_physical(
    logical_coordinates: tuple[int, int],
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
) -> tuple[int, int]:
  """Convert logical coordinates to physical coordinates.

  Args:
    logical_coordinates: The logical coordinates for the point.
    logical_screen_size: The logical screen size.
    physical_frame_boundary: The physical coordinates in portrait orientation
      for the upper left and lower right corner for the frame.
    orientation: The current screen orientation.

  Returns:
    The physical coordinate for the point in portrait orientation.

  Raises:
    ValueError: If the orientation is not valid.
  """
  x, y = logical_coordinates
  px0, py0, px1, py1 = physical_frame_boundary
  px, py = px1 - px0, py1 - py0
  lx, ly = logical_screen_size
  if orientation == 0:
    return (int(x * px / lx) + px0, int(y * py / ly) + py0)
  if orientation == 1:
    return (px - int(y * px / ly) + px0, int(x * py / lx) + py0)
  if orientation == 2:
    return (px - int(x * px / lx) + px0, py - int(y * py / ly) + py0)
  if orientation == 3:
    return (int(y * px / ly) + px0, py - int(x * py / lx) + py0)
  print('Invalid orientation.')
  raise ValueError('Unsupported orientation.')


def _ui_element_logical_corner(
    ui_element: representation_utils.UIElement, orientation: int
) -> list[tuple[int, int]]:
  """Get logical coordinates for corners of a given UI element.

  Args:
    ui_element: The corresponding UI element.
    orientation: The current orientation.

  Returns:
    Logical coordinates for upper left and lower right corner for the UI
    element.

  Raises:
    ValueError: If bounding box is missing.
    ValueError: If orientation is not valid.
  """
  if ui_element.bbox_pixels is None:
    raise ValueError('UI element does not have bounding box.')
  if orientation == 0:
    return [
        (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_min)),
        (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_max)),
    ]
  if orientation == 1:
    return [
        (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_max)),
        (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_min)),
    ]
  if orientation == 2:
    return [
        (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_max)),
        (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_min)),
    ]
  if orientation == 3:
    return [
        (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_min)),
        (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_max)),
    ]
  raise ValueError('Unsupported orientation.')


def add_ui_element_mark(
    screenshot: np.ndarray,
    ui_element: representation_utils.UIElement,
    index: int | str,
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
):
  """Add mark (a bounding box plus index) for a UI element in the screenshot.

  Args:
    screenshot: The screenshot as a numpy ndarray.
    ui_element: The UI element to be marked.
    index: The index for the UI element.
    logical_screen_size: The logical screen size.
    physical_frame_boundary: The physical coordinates in portrait orientation
      for the upper left and lower right corner for the frame.
    orientation: The current screen orientation.
  """
  if ui_element.bbox_pixels:
    upper_left_logical, lower_right_logical = _ui_element_logical_corner(
        ui_element, orientation
    )
    upper_left_physical = _logical_to_physical(
        upper_left_logical,
        logical_screen_size,
        physical_frame_boundary,
        orientation,
    )
    lower_right_physical = _logical_to_physical(
        lower_right_logical,
        logical_screen_size,
        physical_frame_boundary,
        orientation,
    )

    cv2.rectangle(
        screenshot,
        upper_left_physical,
        lower_right_physical,
        color=(0, 255, 0),
        thickness=2,
    )
    screenshot[
        upper_left_physical[1] + 1 : upper_left_physical[1] + 25,
        upper_left_physical[0] + 1 : upper_left_physical[0] + 35,
        :,
    ] = (255, 255, 255)
    cv2.putText(
        screenshot,
        str(index),
        (
            upper_left_physical[0] + 1,
            upper_left_physical[1] + 20,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        thickness=2,
    )


def add_screenshot_label(screenshot: np.ndarray, label: str):
  """Add a text label to the right bottom of the screenshot.

  Args:
    screenshot: The screenshot as a numpy ndarray.
    label: The text label to add, just a single word.
  """
  height, width, _ = screenshot.shape
  screenshot[height - 30 : height, width - 150 : width, :] = (255, 255, 255)
  cv2.putText(
      screenshot,
      label,
      (width - 120, height - 5),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 0, 0),
      thickness=2,
  )


def encode_image_for_html(image: np.ndarray) -> str:
  """Encode image in numpy ndarray to html string with correct color channels.

  Args:
    image: Image as a numpy ndarray.

  Returns:
    Encoded image to be used in html.
  """
  return base64.b64encode(
      cv2.imencode('.jpeg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))[1]
  ).decode('utf-8')


def parse_reason_action_output(
    raw_reason_action_output: str,
) -> tuple[Optional[str], Optional[str]]:
  r"""Parses llm action reason output.

  Args:
    raw_reason_action_output: Raw string output that supposes to have the format
      'Reason: xxx\nAction:xxx'.

  Returns:
    If parsing successfully, returns reason and action.
  """
  reason_result = re.search(
      r'Low-level thought:(.*)action:', raw_reason_action_output, flags=re.DOTALL
  )
  reason = reason_result.group(1).strip() if reason_result else None
  action_result = re.search(
      r'action:(.*)', raw_reason_action_output, flags=re.DOTALL
  )
  action = action_result.group(1).strip() if action_result else None
  return reason, action

# "conversations": [{"from": "human", "value": "<image>\nYou are a GUI task expert, I will provide you with a high-level instruction, a screenshot and its corresponding accessibility tree.\n\nHigh-level instruction: Configure the audio recording settings to M4a format, 44.1kHz sample rate, 128 kbps bitrate, and Stereo channel. Then apply these settings in the Audio Recorder app.\n\nAccessibility tree: {\"com.dimowner.audiorecorder\": \"(540.0, 1085.5)\", \"Get started\": \"(539.5, 2148.0)\", \"Audio Recorder\": \"(539.5, 1158.5)\", \"Welcome to a simple and fast application\": \"(540.0, 1312.0)\", \"15:34\": \"(99.0, 64.5)\", \"SMS Messenger notification: Matteo Ibrahim\": \"(177.0, 64.5)\", \"Clock notification: \": \"(235.0, 64.5)\", \"Phone signal full.\": \"(817.0, 64.0)\", \"Battery 100 percent.\": \"(945.5, 64.5)\", \"No internet\": \"(797.5, 64.0)\"}\n\nPlease generate the low-level thought and action for the next step."}, {"from": "gpt", "value": "Low-level thought: Click 'Get started' to open the setup screen for configuring audio recording settings.\naction: {\"action_type\":\"click\",\"x\":539.5,\"y\":2148.0}"}]}


def _generate_screenshot_table(task_result: dict[str, Any], i: int) -> str:
  """Generate html string for the screenshot analysis table.

  Args:
    task_result: Task run result by M3A.
    i: The index of the step.

  Returns:
    Html string for the screenshot analysis table.
  """
  html_str = (
      "<table style='width:100%;'><caption"
      " style='caption-side:top;text-align:left;'>Screenshot Analysis</caption>"
  )

  # Column for the raw screenshot
  if task_result['episode_data']['raw_screenshot'][i] is not None:
    encoded_raw_screenshot = encode_image_for_html(
        task_result['episode_data']['raw_screenshot'][i]
    )
    html_str += f"""
      <tr>
        <td style='text-align:center;'>
          Before Screenshot (raw):<br>
          <img src="data:image/png;base64,{encoded_raw_screenshot}" alt="Raw Screenshot" width="324" height="720">
        </td>
    """

  # Column for the screenshot before actions with marks
  if task_result['episode_data']['before_screenshot_with_som'][i] is not None:
    encoded_before_screenshot = encode_image_for_html(
        task_result['episode_data']['before_screenshot_with_som'][i]
    )
    html_str += f"""
        <td style='text-align:center;'>
          Before Screenshot with marks:<br>
          <img src="data:image/png;base64,{encoded_before_screenshot}" alt="Before Screenshot with Marks" width="324" height="720">
        </td>
    """

  # Column for the screenshot after actions with marks
  if task_result['episode_data']['after_screenshot_with_som'][i] is not None:
    encoded_after_screenshot = encode_image_for_html(
        task_result['episode_data']['after_screenshot_with_som'][i]
    )
    html_str += f"""
        <td style='text-align:center;'>
          After Screenshot with marks:<br>
          <img src="data:image/png;base64,{encoded_after_screenshot}" alt="After Screenshot with Marks" width="324" height="720">
        </td>
      </tr>
    """

  html_str += '</table>'
  return html_str


def generate_single_task_html_for_m3a(task_result: dict[str, Any]) -> str:
  """Generates html string for a task result obtained by M3A.

  Args:
    task_result: Task run result by M3A.

  Returns:
    Raw html string for this result.
  """
  if np.isnan(task_result['is_successful']):
    return (
        '<p>Some error happened during the execution for this task, no result'
        ' available.</p>'
    )

  html_str = f"""
    Goal: {task_result['goal']}<br>
    Status: {'success' if task_result['is_successful'] else 'fail'}<br>
    Duration: {"{:.3f}".format(task_result['run_time'])} seconds</p>
    """
  n_step = len(task_result['episode_data']['summary'])
  for i in range(n_step):
    reason, action = parse_reason_action_output(
        task_result['episode_data']['action_output'][i]
        if task_result['episode_data']['action_output'][i]
        else 'No output available.'
    )
    html_str += f'<p>Step {str(i)} <br>'
    if reason and action:
      html_str += f"""
          Reason: {reason if reason else 'Output not in correct format.'}<br>
          Action: {action if action else 'Output not in correct format.'}<br>
          """
    else:
      html_str += (
          'Action Selection output not in correct format.<br> Output: '
          + (
              task_result['episode_data']['action_output'][i]
              if task_result['episode_data']['action_output'][i]
              else 'No output available.'
          )
          + '<br>'
      )

    summary = (
        task_result['episode_data']['summary'][i]
        if task_result['episode_data']['summary'][i]
        else 'Summary not available.'
    )
    html_str += f'Summary: {summary}</p>'
    html_str += _generate_screenshot_table(task_result, i)
  return html_str


def generate_eval_html_report(
    task_results: list[dict[str, Any]], agent_type: str, fail_only: bool = False
) -> str:
  """Generate evaluation results report as a html string.

  Notice that the task_results MUST be obtained by the suite_utils.run function
  (or loaded using Checkpointer) with one of the supported agent type.

  Sample usage:
    # import webbrowser
    # agent = m3a.M3A(...)
    # task_results1 = suite_utils.run(suite, env, agent)
    #
    # result_path = 'xxx'
    # raw_result_checkpoint = checkpointer_lib.Checkpointer(result_path)
    # task_results2, _ = raw_result_checkpoint.load()
    #
    # output_path = xxx
    # with open(output_path, 'wb') as f:
    #   f.write(generate_eval_html_report(
    #       task_results1, # Or task_results2
    #       agent.__class__.__name__,
    #       False)
    #   )
    # webbrowser.open_new_tab(output_path)

  Args:
    task_results: List of task results obtained by running the suite_utils's run
      function with the agent.
    agent_type: Indicate which agent generate the task_results above.
    fail_only: Indicate if the report should only contain failed cases.

  Returns:
    Html string for the result report.
  """
  if agent_type == 'M3A':
    single_result_html_generation = generate_single_task_html_for_m3a
  elif agent_type == 'T3A':
    single_result_html_generation = generate_single_task_html_for_gpt4_text
  else:
    print('Currently only supports results obtained by M3A or T3A.')
    raise ValueError('Unsupported agent type.')

  html_str = (
      '<html><body style="word-wrap: break-word; background-color: #d9ead3;">'
  )

  for index, task_result in enumerate(task_results):
    if (
        fail_only
        and isinstance(task_result['is_successful'], bool)
        and task_result['is_successful']
    ):
      continue
    html_str += (
        f'<p>===============================<br>Task {str(index+1)}:'
        f' {task_result["task_template"]}<br>'
        + single_result_html_generation(task_result)
    )
  html_str += '</body></html>'
  return html_str


def generate_single_task_html_for_gpt4_text(task_result: dict[str, Any]) -> str:
  """Generates html string for a task result obtained by Gpt4TextAgent.

  Args:
    task_result: Task run result by Gpt4TextAgent.

  Returns:
    Raw html string for this result.
  """
  if np.isnan(task_result['is_successful']):
    return (
        '<p>Some error happened during the execution for this task, no result'
        ' available.</p>'
    )

  html_str = f"""
    Goal: {task_result['goal']}<br>
    Status: {'success' if task_result['is_successful'] else 'fail'}<br>
    Duration: {"{:.3f}".format(task_result['run_time'])} seconds</p>
    """
  n_step = len(task_result['episode_data']['summary'])
  for i in range(n_step):
    reason, action = parse_reason_action_output(
        task_result['episode_data']['action_output'][i]
    )
    html_str += f"""
      <p>Step {str(i)} <br>
      Reason: {reason}<br>
      Action: {action}<br>
      Summary: {task_result['episode_data']['summary'][i]}</p>
      """
    if task_result['episode_data']['before_screenshot'][i] is not None:
      encoded_before_screenshot = encode_image_for_html(
          task_result['episode_data']['before_screenshot'][i]
      )
      html_str += f"""
        Before Screenshot:
        <img src="data:image/png;base64,{encoded_before_screenshot}" alt="Image" width="324" height="720">
        """
    if task_result['episode_data']['after_screenshot'][i] is not None:
      encoded_after_screenshot = encode_image_for_html(
          task_result['episode_data']['after_screenshot'][i]
      )
      html_str += f"""
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        After Screenshot:
        <img src="data:image/png;base64,{encoded_after_screenshot}" alt="Image" width="324" height="720">
        """
  return html_str


def validate_ui_element(
    ui_element: representation_utils.UIElement,
    screen_width_height_px: tuple[int, int],
) -> bool:
  """Used to filter out invalid UI element."""
  screen_width, screen_height = screen_width_height_px

  # Filters out invisible element.
  if not ui_element.is_visible:
    return False

  # Filters out element with invalid bounding box.
  if ui_element.bbox_pixels:
    x_min = ui_element.bbox_pixels.x_min
    x_max = ui_element.bbox_pixels.x_max
    y_min = ui_element.bbox_pixels.y_min
    y_max = ui_element.bbox_pixels.y_max

    if (
        x_min >= x_max
        or x_min >= screen_width
        or x_max <= 0
        or y_min >= y_max
        or y_min >= screen_height
        or y_max <= 0
    ):
      return False

  return True
