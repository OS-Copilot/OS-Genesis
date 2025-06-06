import os
import random
import requests
import time
from PIL import Image
import base64
import io
import json
import re
from tqdm import tqdm
import numpy as np


def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')


def convert_image_to_base64(image_path):
    # Open the image file
    with open(image_path, 'rb') as f:
        # Load the image using PIL
        image_bytes = f.read()
        # Encode the image bytes to base64
        encoded_image = encode_image(image_bytes)
        return encoded_image


def call_llm(model_name, payload):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }
    print("Generating content with GPT model: {}".format(model_name))
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        if response.json()['error']['code'] == "context_length_exceeded":
            print("Context length exceeded. Retrying with a smaller context.")
            payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
            retry_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            if retry_response.status_code != 200:
                print(
                    "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                return ""

        print("Failed to call LLM: " + response.text)
        time.sleep(2)
        return ""
    else:
        return response.json()['choices'][0]['message']['content']


gpt_annot_traj = json.load(open('/Users/cckevin/Desktop/gpt_annot_traj_v2.json', 'r'))
imgs_dir = '/Users/cckevin/Desktop/gpt_annot_traj_v2/gpt_screenshots_v2'

system_prompt = """
You are an expert in evaluating Android GUI agent task trajectories. Your task is to assess the quality and effectiveness of task trajectories for GUI manipulation tasks. 

A trajectory consists of the following components:
1. High-level Instruction: Describes the user's intended task (e.g., "Create a new travel itinerary document in a folder").
2. Action History: Includes two key parts:
   - Low-level Actions & Summaries: A sequence of actions, where each step includes:
     - The executed action.
     - A summary of the action, indicating the effect after the action is executed.
   - GUI Screenshots: Screenshots captured when the last three actions are executed: the third-to-last, second-to-last, and final actions (if there are at least three actions; otherwise, include all actions).

When evaluating a trajectory, consider these key aspects:

### Evaluation Criteria:
1. Trajectory Coherence:
   - Do the low-level steps and corresponding actions follow a logical sequence toward the goal?
   - Are the actions clearly described and specific?
   - Are there redundant or unnecessary actions?

2. Task Completion:
   - Does the trajectory successfully achieve the instructed task?
   - Are all necessary interactions completed?
   - Are error cases handled appropriately?

### Scoring Guidelines:
Rate the trajectory on a scale of 1 to 5 based on the evaluation criteria:

- 5: The task is perfectly completed, successfully executing multiple actions to achieve the goal. The sequence is logically clear with no noticeable redundancies.
- 4: The task is mostly completed, successfully executing multiple actions. However, due to challenges or ambiguities in the instructions, the completion is not perfect, or there are inefficiencies in the process.
- 3: The task is partially completed, with some successful actions executed. However, due to task or environmental constraints, the goal is not fully achieved, or the sequence ends in a loop or error.
- 2: Only a few actions are executed. Although there is an attempt to complete the task, the trajectory deviates from the goal early on or demonstrates significant inefficiencies in execution and logic.
- 1: The task fails completely, with no meaningful actions executed at the start. The sequence either falls into an immediate deadlock, a repetitive loop, or demonstrates no value in completing the task.

Note: If the task is relatively complex, but the trajectory demonstrates valuable attempts, even if the task is not fully completed, consider adjusting the score upward. However, if the task is complex but the trajectory fails to perform actions that contribute meaningfully to task completion, no extra points should be awarded.

### Response Format:
Format your response into two lines as shown below: 
Reason: <your thoughts and reasoning process for the score>
Score: <your score from 1-5>
"""

for i, item in enumerate(gpt_annot_traj[:]):

    if "reward" in item:
        print("processed")
        continue

    instruction = item["instruction"]

    action_history = []
    for j, action in enumerate(item["steps"]):
        if "summary" in action:
            summary = action["summary"]
        else:
            summary = action["reason"]
        summary = f"Step {j+1}: {summary}"
        action_history.append(summary)
    action_history_text = '\n'.join(action_history)

    action_screenshots = []
    for action in item["steps"][-3:]:
        screenshot_path = os.path.join(imgs_dir, action["screen_before"])
        screenshot = convert_image_to_base64(screenshot_path)
        action_screenshots.append(screenshot)

    traj_prompt = f"Instruction :{instruction}\nAction History:\n{action_history_text}\nThe last three screenshots are provided."

    messages = []

    messages.append({
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt
            },
        ]
    })

    # Prediction example
    action_text_image = []
    for img in action_screenshots:
        action_text_image.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img}",
                    "detail": "high"
                }
            }
        )

    action_text_image.append(
        {
            "type": "text",
            "text": traj_prompt
        }
    )

    messages.append({
        "role": "user",
        "content": action_text_image
    })

    print(traj_prompt)

    model_name = "gpt-4o-2024-08-06"
    try_num = 0
    answer = None
    while try_num < 5:
        try_num += 1
        try:
            response = call_llm(model_name, {
                "model": model_name,
                "messages": messages,
                "max_tokens": 1500,
                "top_p": 0.9,
                "temperature": 0.5
            })
        except:
            print("error call")
            time.sleep(1.0)
            continue
        try:
            print(response)
            reason_match = re.search(r"Reason:\s*(.+?)\s*Score:", response, re.DOTALL)
            score_match = re.search(r"Score:\s*(\d+)", response)
            reason = reason_match.group(1).strip() if reason_match else None
            score = int(score_match.group(1)) if score_match else None

            if reason and score and 1 <= score <= 5:
                item["reward_reason"] = reason
                item["reward"] = score
                break  # Successfully parsed, exit loop
            else:
                print("Invalid response format or score out of range, retrying...")
                time.sleep(1.0)

        except json.JSONDecodeError:
            # If response is not valid JSON, continue to generate
            print("Invalid response received, retrying...")
            time.sleep(1.0)

    num_processed = len([item for item in gpt_annot_traj if ("reward" in item)])
    print("Num of total: {} Num of success: {}".format(len(gpt_annot_traj), num_processed))
    if i % 20 == 0:
        json.dump(gpt_annot_traj, open('/Users/cckevin/Desktop/gpt_annot_traj_v2_reward.json', 'w'))

json.dump(gpt_annot_traj, open('/Users/cckevin/Desktop/gpt_annot_traj_v2_reward.json', 'w'))
print("Save")