"""OS-Genesis Agents for Android (adapted from M3A)."""

import time
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import genesis_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
import json
import re
from android_world.agents.t3a import _generate_ui_elements_description_list_full

PROMPT_PREFIX = (
    'You are a GUI task expert, I will provide you with a high-level instruction, an action history, a screenshot with its corresponding accessibility tree.'
)

# Updated ACTION_SELECTION_PROMPT_TEMPLATE to match the new format
ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\n\nHigh-level instruction: {goal}\n\n'
    + 'Action history: {history}\n\n'
    + 'Accessibility tree: {accessibility_tree}\n\n'
    + 'Please generate the low-level thought and action for the next step.'
    # + GUIDANCE
)

SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is: {goal}\n'
    'Now I want you to summerize the latest step.\n'
    'You will be given the screenshot before you performed the action (which'
    ' has a text label "before" on the bottom right), the action you chose'
    ' (together with the reason) and the screenshot after the action was'
    ' performed (which has a text label "after" on the bottom right).\n'
    'Also here is the list of detailed information for some UI elements'
    ' in the before screenshot:\n{before_elements}\n'
    'Here is the list for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    'By comparing the two screenshots (plus the UI element lists) and the'
    ' action performed, give a brief summary of this step. This summary'
    ' will be added to action history and used in future action selection,'
    ' so try to include essential information you think that will be most'
    ' useful for future action selections like what you'
    ' intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might be'
    ' wrong), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short (better less than 50 words) and in a single line\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)


def _generate_ui_element_description(
    ui_element: representation_utils.UIElement, index: int
) -> str:
  """Generate a description for a given UI element with important information.

  Args:
    ui_element: UI elements for the current screen.
    index: The numeric index for the UI element.

  Returns:
    The description for the UI element.
  """
  element_description = f'UI element {index}: {{"index": {index}, '
  if ui_element.text:
    element_description += f'"text": "{ui_element.text}", '
  if ui_element.content_description:
    element_description += (
        f'"content_description": "{ui_element.content_description}", '
    )
  if ui_element.hint_text:
    element_description += f'"hint_text": "{ui_element.hint_text}", '
  if ui_element.tooltip:
    element_description += f'"tooltip": "{ui_element.tooltip}", '
  element_description += (
      f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
  )
  element_description += (
      '"is_long_clickable":'
      f' {"True" if ui_element.is_long_clickable else "False"}, '
  )
  element_description += (
      f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
  )
  if ui_element.is_scrollable:
    element_description += '"is_scrollable": True, '
  if ui_element.is_focusable:
    element_description += '"is_focusable": True, '
  element_description += (
      f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
  )
  element_description += (
      f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
  )
  return element_description[:-2] + '}'


def _generate_ui_elements_description_list(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate concise information for a list of UIElement.

  Args:
    ui_elements: UI elements for the current screen.
    screen_width_height_px: The height and width of the screen in pixels.

  Returns:
    Concise information for each UIElement.
  """
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if genesis_utils.validate_ui_element(ui_element, screen_width_height_px):
      tree_info += _generate_ui_element_description(ui_element, index) + '\n'
  return tree_info


def _action_selection_prompt(
    goal: str,
    history: list[str],
    ui_elements: str,
    accessibility_tree: str,
    additional_guidelines: list[str] | None = None,
) -> str:
  """Generate the prompt for the action selection.

  Args:
    goal: The current goal.
    history: Summaries for previous steps.
    ui_elements: A list of descriptions for the UI elements.
    additional_guidelines: Task specific guidelines.

  Returns:
    The text prompt for action selection that will be sent to gpt4v.
  """
  if history:
    history = '\n'.join(history)
  else:
    history = ''

  extra_guidelines = ''
  if additional_guidelines:
    extra_guidelines = 'For The Current Task:\n'
    for guideline in additional_guidelines:
      extra_guidelines += f'- {guideline}\n'

  return ACTION_SELECTION_PROMPT_TEMPLATE.format(
      goal=goal,
      history=history,
      ui_elements=ui_elements if ui_elements else 'Not available',
      additional_guidelines=extra_guidelines,
      accessibility_tree=accessibility_tree
  )


def _summarize_prompt(
    action: str,
    reason: str,
    goal: str,
    before_elements: str,
    after_elements: str,
) -> str:
  """Generate the prompt for the summarization step.

  Args:
    action: Action picked.
    reason: The reason to pick the action.
    goal: The overall goal.
    before_elements: Information for UI elements on the before screenshot.
    after_elements: Information for UI elements on the after screenshot.

  Returns:
    The text prompt for summarization that will be sent to gpt4v.
  """
  return SUMMARY_PROMPT_TEMPLATE.format(
      goal=goal,
      before_elements=before_elements,
      after_elements=after_elements,
      action=action,
      reason=reason,
  )


def _generate_a11y_tree_ours(env_state, logical_screen_size):
    def element_to_identifier(element):
        """Converts an element to a JSON-serializable identifier."""
        bbox = getattr(element, 'bbox_pixels', None)
        bbox_dict = {'x_min': bbox.x_min, 'x_max': bbox.x_max, 'y_min': bbox.y_min,
                     'y_max': bbox.y_max} if bbox else None
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

    # def get_ui_elements_clean(ui_elements):
    #     # 从AW保存的ui_elements表示中过滤得到clean的表示
    #     ui_elements_clean = []
    #     for ele in ui_elements:
    #         if ele["text"] != None or ele["text"] == "":
    #             ele_text = ele["text"]
    #         elif ele["content_description"] != None or ele["content_description"] == "":
    #             ele_text = ele["content_description"]
    #         else:
    #             ele_text = ele["package_name"]

    #         if (not ele_text) or ele_text == "":
    #             print("text not exist")
    #             input()

    #         bbox = ele["bbox_pixels"]
    #         ele_point = [(bbox["x_min"] + bbox["x_max"]) / 2, (bbox["y_min"] + bbox["y_max"]) / 2]

    #         ui_elements_clean.append({"text": ele_text, "center": ele_point})

    #     return ui_elements_clean
    def get_ui_elements_clean(ui_elements, ui_elements_text):
        def extract_is_checked(input_string, element_number):
            # 使用正则表达式匹配指定UI element
            element_pattern = rf"UI element {element_number}:.*?(?=UI element|$)"
            element_match = re.search(element_pattern, input_string, re.DOTALL)
            if not element_match:
                raise ValueError("not found")

            # 获取指定UI element的内容
            element_content = element_match.group()

            # 检查is_checked的值
            is_checked_pattern = r"is_checked=(True|False)"
            is_checked_match = re.search(is_checked_pattern, element_content)
            if not is_checked_match:
                raise ValueError("not found")

            return is_checked_match.group(1)  # 返回True或False

        # 从AW保存的ui_elements表示中过滤得到clean的表示
        ui_elements_clean = []
        for i, ele in enumerate(ui_elements):
            if ele["text"] != None:
                ele_text = ele["text"]
            elif ele["content_description"] != None:
                ele_text = ele["content_description"]
            else:  # 如果text和content_description属性不存在，用其余几个重要属性表示
                if ele["class_name"] == "android.widget.Switch":
                    ele_text = ele["class_name"]
                else:
                    names = [ele["class_name"], ele["package_name"], ele["resource_name"]]
                    names = [item for item in names if item]
                    ele_text = ' '.join(names)

            if (not ele_text) or ele_text == "":
                print("text not exist")
                input()

            # 如果是开关类元素，补充checked状态
            if "android.widget.Switch" in ele_text:
                # print('widget_switch')
                is_checked = extract_is_checked(ui_elements_text, i)    # 从element_text中找is_checked
                ele_text = f"{ele_text} Is_Checked: {is_checked}"

            bbox = ele["bbox_pixels"]
            ele_point = [(bbox["x_min"]+bbox["x_max"])/2, (bbox["y_min"]+bbox["y_max"])/2]

            ui_elements_clean.append({"text": ele_text, "center": ele_point})

        return ui_elements_clean


    ui_elements = env_state.ui_elements
    ui_elements_before_identifiers = [element_to_identifier(elem) for elem in ui_elements if
                                      genesis_utils.validate_ui_element(elem, logical_screen_size)]
    element_list_text = _generate_ui_elements_description_list_full(
        ui_elements,
        logical_screen_size,
    )
    ui_elements_before_clean = get_ui_elements_clean(ui_elements_before_identifiers, element_list_text)
    a11y_tree = {}
    for item in ui_elements_before_clean:
        text = item["text"]
        point = "({}, {})".format(item["center"][0], item["center"][1])
        a11y_tree[text] = point
    a11y_tree_ours = json.dumps(a11y_tree, ensure_ascii=False)
    return a11y_tree_ours, ui_elements_before_identifiers


def reverse_action(action_json, ui_elements_before):
    action_type = action_json.get("action_type")

    # 查找坐标最接近的元素的index
    def find_closest_element_index(x, y, ui_elements_before):
        closest_index = None
        min_distance = float('inf')

        # 遍历所有UI元素，计算与给定坐标的距离
        for idx, element in enumerate(ui_elements_before):
            bbox = element["bbox_pixels"]
            # 计算元素的中心坐标
            element_center_x = (bbox["x_min"] + bbox["x_max"]) / 2
            element_center_y = (bbox["y_min"] + bbox["y_max"]) / 2
            # 计算坐标之间的欧式距离
            distance = ((x - element_center_x) ** 2 + (y - element_center_y) ** 2) ** 0.5
            # 寻找最小的距离
            if distance < min_distance:
                min_distance = distance
                closest_index = idx

        return closest_index

    if action_type == "click":
        # 提取坐标
        x = action_json.get("x")
        y = action_json.get("y")
        # 查找与坐标最接近的元素
        closest_index = find_closest_element_index(x, y, ui_elements_before)
        if closest_index is not None:
            return {"action_type": "click", "index": closest_index}
        else:
            print("No matching element found for the given coordinates.")
            return None

    elif action_type == "type":
        # 提取输入文本和坐标
        typed_text = action_json.get("text")
        x = action_json.get("x")
        y = action_json.get("y")
        # 查找与坐标最接近的元素
        closest_index = find_closest_element_index(x, y, ui_elements_before)
        if closest_index is not None:
            return {"action_type": "input_text", "index": closest_index, "text": typed_text}
        else:
            print("No matching element found for the given coordinates.")
            return None

    elif action_type == "long_press":
        # 提取坐标
        x = action_json.get("x")
        y = action_json.get("y")
        # 查找与坐标最接近的元素
        closest_index = find_closest_element_index(x, y, ui_elements_before)
        if closest_index is not None:
            return {"action_type": "long_press", "index": closest_index}
        else:
            print("No matching element found for the given coordinates.")
            return None

    elif action_type == "scroll":
        # 直接返回方向
        direction = action_json.get("direction")
        if direction:
            return {"action_type": "scroll", "direction": direction}
        else:
            print("Invalid scroll direction")
            return None

    elif action_type == "navigate_back":
        return {"action_type": "navigate_back"}

    elif action_type == "navigate_home":
        return {"action_type": "navigate_home"}

    elif action_type == "keyboard_enter":
        return {"action_type": "keyboard_enter"}

    elif action_type == "wait":
        return {"action_type": "wait"}

    elif action_type == "open_app":
        # 提取app名称
        app_name = action_json.get("app_name")
        if app_name:
            return {"action_type": "open_app", "app_name": app_name}
        else:
            print("Invalid app name")
            return None

    elif action_type == "status":
        goal_status = action_json.get("goal_status")
        if goal_status == "successful":
            return {"action_type": "status", "goal_status": "complete"}
        elif goal_status == "infeasible":
            return {"action_type": "status", "goal_status": "infeasible"}
        else:
            print("Invalid goal status")
            return None
        
    elif action_type == "answer":
        answer_text = action_json.get("text")
        return {"action_type": "answer", "text": answer_text}

    else:
        print(f"Unknown action type: {action_type}")
        return None


class M3A(base_agent.EnvironmentInteractingAgent):
  """M3A which stands for Multimodal Autonomous Agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      name: str = 'M3A',
      wait_after_action_seconds: float = 2.0,
  ):
    """Initializes a M3A Agent.

    Args:
      env: The environment.
      llm: The multimodal LLM wrapper.
      name: The agent name.
      wait_after_action_seconds: Seconds to wait for the screen to stablize
        after executing an action
    """
    super().__init__(env, name)
    self.llm = llm
    self.history = []
    self.additional_guidelines = None
    self.wait_after_action_seconds = wait_after_action_seconds

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    # Hide the coordinates on screen which might affect the vision model.
    self.env.hide_automation_ui()
    self.history = []

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = {
        'raw_screenshot': None,
        'before_screenshot_with_som': None,
        'before_ui_elements': [],
        'after_screenshot_with_som': None,
        'action_prompt': None,
        'action_output': None,
        'action_output_json': None,
        'action_reason': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(len(self.history) + 1))

    state = self.get_post_transition_state()
    logical_screen_size = self.env.logical_screen_size

    a11y_tree_ours, ui_elements_before_idf = _generate_a11y_tree_ours(state, logical_screen_size)

    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary

    before_ui_elements = state.ui_elements
    step_data['before_ui_elements'] = before_ui_elements
    before_ui_elements_list = _generate_ui_elements_description_list(
        before_ui_elements, logical_screen_size
    )
    step_data['raw_screenshot'] = state.pixels.copy()
    before_screenshot = state.pixels.copy()
    for index, ui_element in enumerate(before_ui_elements):
        if genesis_utils.validate_ui_element(ui_element, logical_screen_size):
            genesis_utils.add_ui_element_mark(
                before_screenshot,
                ui_element,
                index,
                logical_screen_size,
                physical_frame_boundary,
                orientation,
            )
    step_data['before_screenshot_with_som'] = before_screenshot.copy()

    # Modify the history construction here
    action_prompt = _action_selection_prompt(
        goal,
        [
            'Step ' + str(i + 1) + ': ' + step_info['summary'] +
            (' action: ' + step_info['action'] if 'action' in step_info else '')
            for i, step_info in enumerate(self.history)
        ],
        before_ui_elements_list,
        a11y_tree_ours,
        self.additional_guidelines,
    )
    step_data['action_prompt'] = action_prompt
    print(action_prompt)
    action_output, is_safe, raw_response = self.llm.predict_mm(
        action_prompt,
        [
            step_data['raw_screenshot'],
            # before_screenshot, 
        ],
    )

    print('Raw Response:', raw_response.json()) # testing
    if is_safe == False:  # pylint: disable=singleton-comparison
        #  is_safe could be None
        action_output = """Reason: Triggered LLM safety classifier.
Action: {"action_type": "status", "goal_status": "infeasible"}"""

    if not raw_response:
        raise RuntimeError('Error calling LLM in action selection phase.')
    step_data['action_output'] = action_output
    step_data['action_raw_response'] = raw_response

    reason, action = genesis_utils.parse_reason_action_output(action_output)

    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    if (not reason) or (not action):
        print('Action prompt output is not in the correct format.')
        step_data['summary'] = (
            'Output for action selection is not in the correct format, so no'
            ' action is performed.'
        )
        self.history.append(step_data)

        return base_agent.AgentInteractionResult(
            False,
            step_data,
        )

    print('Action: ' + action)
    print('Reason: ' + reason)
    step_data['action_reason'] = reason
    step_data['action'] = action
    # qiushi：注意这里为了和训练时候一致，step data里存的是转换之前的坐标形式action

    # convert the parsed action to AndroidWorld format
    action_aw = reverse_action(json.loads(action), ui_elements_before_idf)
    if action_aw == None:
        raise RuntimeError('Error reversing action.')
    action = json.dumps(action_aw, ensure_ascii=False)

    # Store the action in step_data
    

    try:
        converted_action = json_action.JSONAction(
            **agent_utils.extract_json(action),
        )
        step_data['action_output_json'] = converted_action
    except Exception as e:  # pylint: disable=broad-exception-caught
        print('Failed to convert the output to a valid action.')
        print(str(e))
        step_data['summary'] = (
            'Can not parse the output to a valid action. Please make sure to pick'
            ' the action from the list with required parameters (if any) in the'
            ' correct JSON format!'
        )
        self.history.append(step_data)

        return base_agent.AgentInteractionResult(
            False,
            step_data,
        )

    action_index = converted_action.index
    num_ui_elements = len(before_ui_elements)
    if (
        converted_action.action_type
        in ['click', 'long_press', 'input_text', 'scroll']
        and action_index is not None
    ):
        if action_index >= num_ui_elements:
            print(
                f'Index out of range, prediction index is {action_index}, but the'
                f' UI element list only has {num_ui_elements} elements.'
            )
            step_data['summary'] = (
                'The parameter index is out of range. Remember the index must be in'
                ' the UI element list!'
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        # Add mark to the target element.
        genesis_utils.add_ui_element_mark(
            step_data['raw_screenshot'],
            before_ui_elements[action_index],
            action_index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

    if converted_action.action_type == 'status':
        if converted_action.goal_status == 'infeasible':
            print('Agent stopped since it thinks mission impossible.')
        step_data['summary'] = reason
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(
            True,
            step_data,
        )

    if converted_action.action_type == 'answer':
        print('Agent answered with: ' + converted_action.text)

    try:
        self.env.execute_action(converted_action)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print('Failed to execute action.')
        print(str(e))
        step_data['summary'] = (
            'Can not execute the action, make sure to select the action with'
            ' the required parameters (if any) in the correct JSON format!'
        )
        return base_agent.AgentInteractionResult(
            False,
            step_data,
        )

    step_data['summary'] = reason

    time.sleep(self.wait_after_action_seconds)

    self.history.append(step_data)
    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )
