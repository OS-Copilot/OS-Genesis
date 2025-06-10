#!/usr/bin/env python3
# type: ignore

"""
Random walk in WebArena environment, collecting <screen_pre, action, screen_after> triples.
Maintain unclickable element pool, clickable element pool, explored element pool, and implement exploration of the website through heuristic rules.
"""

import json
import os
import random
import re
import subprocess
import time
import copy
import numpy as np
import uuid
from PIL import Image
import requests
from tqdm import tqdm
from browser_env.utils import (
    DetachedPage,
)


def get_state(env):
    """Get the current state of the environment"""
    observation = env._get_obs()
    observation_metadata = env._get_obs_metadata()
    info = {
        "page": DetachedPage(env.page.url, ""),
        "fail_error": "",
        "observation_metadata": observation_metadata,
    }
    # Return a deep copy of info to ensure it doesn't update with the environment
    return (copy.deepcopy(observation), copy.deepcopy(info))


def extract_state_ele(observation_metadata):
    # Extract each element represented by 'union_bound' and 'text' from the current interface for comparing changes between interfaces
    return [(tuple(value.get('union_bound')), re.sub(r'^\[\d+\]\s*', '', value.get('text'))) for value in observation_metadata.values()]


def are_screen_identical(screen_before, screen_after):
    return np.array_equal(screen_before, screen_after)


def load_element_pool(file_path):
    """Load element pool"""
    if not os.path.exists(file_path):
        return set()
    else:
        with open(file_path, 'r') as f:
            data_list = json.load(f)
            element_pool = set((tuple(item[0]), item[1]) for item in data_list)
            return element_pool


def save_element_pool(element_pool, file_path):
    """Save element pool"""
    with open(file_path, 'w') as f:
        json.dump(list(element_pool), f, indent=2)


def save_image(image_array, directory):
    """Save image and return filename"""
    unique_id = str(uuid.uuid4())
    image_name = f"{unique_id}.png"
    image_path = os.path.join(directory, image_name)
    image = Image.fromarray(image_array)
    image.save(image_path)
    return image_name


def select_element(state_elements, actree_obs, unclick_elem_pool, new_elements, explored_elem_pool):
    """Randomly select an element based on weights and return corresponding information"""
    elements_weights = []
    for action_element_id, action_element in state_elements:
        element_key = (tuple(action_element['union_bound']), re.sub(r'^\[\d+\]\s*', '', action_element['text']))
        # Check if element is in unclickable element set and if it's visible in the interface
        if element_key in unclick_elem_pool or (f"[{action_element_id}]" not in actree_obs) or ("statictext" in element_key[1].lower()):
            continue
        # Determine element status
        is_new = element_key in new_elements
        is_explored = element_key in explored_elem_pool
        # Assign weights
        if (not is_explored) and is_new:
            weight = 4  # Unexplored and newly appeared, highest weight
        elif is_explored and is_new:
            weight = 3  # Explored but newly appeared, second highest weight
        elif (not is_explored) and (not is_new):
            weight = 3  # Unexplored but not newly appeared, second highest weight
        else:
            weight = 1  # Explored and not newly appeared, lowest weight
        elements_weights.append((action_element_id, action_element, element_key, weight))

    # If no selectable elements, return None
    if len(elements_weights) == 0:
        return None

    print(elements_weights)

    # Randomly select an element based on weights
    elem_infos = [item[:3] for item in elements_weights]
    weights = [item[3] for item in elements_weights]
    selected_elem_info = random.choices(elem_infos, weights=weights, k=1)[0]
    return selected_elem_info


def generate_text_input(screen_text, selected_element_text, max_retries=5):
    """Call GPT API to determine if element is an input field and generate input content"""
    def is_valid_response(response_text):
        try:
            # Clean up the response text
            response_text = response_text.strip()

            # Remove code block markers if present
            if response_text.startswith("```"):
                # Remove starting ``` or ```json
                response_text = re.sub(r'^```[a-zA-Z]*\n', '', response_text)
                # Remove ending ```
                response_text = re.sub(r'\n```$', '', response_text)

            # Extract JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Replace single quotes with double quotes
                json_str = json_str.replace("'", '"')
                # Replace True/False with true/false
                json_str = json_str.replace('True', 'true').replace('False', 'false')
                response_json = json.loads(json_str)
                # Check if 'is_input' exists and is boolean
                if 'is_input' in response_json and isinstance(response_json['is_input'], bool):
                    if response_json['is_input']:
                        # Ensure 'input_content' exists and is a string
                        return 'input_content' in response_json and isinstance(response_json['input_content'], str)
                    else:
                        return True
                else:
                    return False
            else:
                return False
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            return False

    prompt = f"""
    You are an intelligent assistant. Based on the current UI elements and the selected element information, carefully determine whether the selected element is an input field. If it is, generate text content that a user might input into this element. The text should be contextually appropriate, for example, if it's a search box, you might generate a search query; if it's a username input field, you might generate a username; if it's a location, you might try the name of a university, museum, airport, etc. Use imagination to generate diversed and appropriate input content.
    
    Current UI elements:
    {screen_text}
    
    Selected element information:
    {selected_element_text}
    
    Please output a JSON object in the following format, without adding any extra text or comments:
    
    If the selected element is an input field:
    
    {{
        "is_input": true,
        "input_content": "the content to input"
    }}
    
    If the selected element is not an input field:
    
    {{
        "is_input": false
    }}
    
    Ensure that the JSON is properly formatted and parsable. Use lowercase `true` or `false` for boolean values, and double quotes for strings.
    
    If you understand, please provide the JSON object now, without adding any extra text or markers.
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
        "max_tokens": 100,
        "temperature": 0.9,
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
            response_text = result['choices'][0]['message']['content'].strip()

            # Validate and parse the response
            if is_valid_response(response_text):
                response_json = json.loads(response_text)
                return response_json
            else:
                print(f"Invalid response format: '{response_text}'. Retrying...")
                retries += 1
                time.sleep(1)  # Wait a bit before retrying
        except Exception as e:
            print(f"Error generating text input: {e}")
            retries += 1
            time.sleep(1)

    # If all retries fail, return default action
    print("Failed to get valid response after retries. Assuming it's not an input field.")
    return {"is_input": False}


def save_trajectory(unique_id_traj, trajectory, traj_dir, screen_dir, website_name, url):
    """Save trajectory data"""
    traj_save = []
    for i, item in enumerate(trajectory):
        if isinstance(item, dict):
            continue
        elif isinstance(item, tuple) and i+1 < len(trajectory):
            screen_before_name = save_image(trajectory[i-1]['observation']['image'], screen_dir)
            screen_after_name = save_image(trajectory[i+1]['observation']['image'], screen_dir)
            step_data = {
                "website_name": website_name,
                "url": url,
                "screen_before": screen_before_name,
                "a11y_before": trajectory[i-1]['observation']['text'],
                "state_before": trajectory[i-1]['info']['observation_metadata'],
                "screen_after": screen_after_name,
                "a11y_after": trajectory[i+1]['observation']['text'],
                "state_after": trajectory[i+1]['info']['observation_metadata'],
                "action": item
            }
            traj_save.append(step_data)

    traj_filename = f"{unique_id_traj}.json"
    traj_path = os.path.join(traj_dir, traj_filename)
    with open(traj_path, 'w') as f:
        json.dump(traj_save, f, indent=2)


def random_walk_episode(config_file, traj_dir, screen_dir, element_dir, config_randomwalk):
    """
    config_file: Initialize a specific website
    """

    if not os.path.exists(traj_dir):
        os.mkdir(traj_dir)
    if not os.path.exists(screen_dir):
        os.mkdir(screen_dir)
    if not os.path.exists(element_dir):
        os.mkdir(element_dir)
    if not os.path.exists(config_randomwalk):
        os.mkdir(config_randomwalk)

    SLEEP = 3
    # set the URLs of each website
    os.environ[
        "SHOPPING"
    ] = "http://ec2-18-220-173-105.us-east-2.compute.amazonaws.com:7770"
    os.environ[
        "SHOPPING_ADMIN"
    ] = "http://ec2-18-220-173-105.us-east-2.compute.amazonaws.com:7780/admin"
    os.environ[
        "REDDIT"
    ] = "http://ec2-18-220-173-105.us-east-2.compute.amazonaws.com:9999"
    os.environ[
        "GITLAB"
    ] = "http://ec2-18-220-173-105.us-east-2.compute.amazonaws.com:8023"
    os.environ[
        "MAP"
    ] = "http://ec2-18-220-173-105.us-east-2.compute.amazonaws.com:3000"
    os.environ[
        "WIKIPEDIA"
    ] = "http://ec2-18-220-173-105.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
    os.environ[
        "HOMEPAGE"
    ] = "PASS"  # The home page is not currently hosted in the demo site
    print("Done setting up URLs")


    # Check if configuration file is correct
    assert os.path.exists(config_file)
    with open(config_file, "r") as f:
        config = json.load(f)

    # Check which URLs are available for the corresponding website and randomly select one
    configs_dir = "/Users/cckevin/Desktop/webarena/config_files"
    website_url = dict()
    for config_item in os.listdir(configs_dir):
        if config_item == "examples":
            continue
        config_path = os.path.join(configs_dir, config_item)
        config_item = json.load(open(config_path, 'r'))
        if not isinstance(config_item, dict):
            continue
        try:
            if len(config_item["sites"]) != 1:
                continue
            website_name = config_item["sites"][0]
            if website_name not in website_url:
                website_url[website_name] = set()

            start_url = config_item["start_url"]
            website_url[website_name].add(start_url)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(config_item)

    web_urls = website_url[config["sites"][0]]
    random_url = random.choice(list(web_urls))
    print(f"Random URL: {random_url}")
    config["start_url"] = random_url
    # Save config
    unique_id_traj = str(uuid.uuid4())
    config_randomwalk_path = os.path.join(config_randomwalk, f"{unique_id_traj}.json")
    json.dump(config, open(config_randomwalk_path, 'w'))

    """
    # run bash prepare.sh to save all account cookies, this only needs to be done once
    subprocess.run(["bash", "prepare.sh"])
    print("Done saving account cookies")
    """

    # Init an environment
    from browser_env import (
        Action,
        ActionTypes,
        ObservationMetadata,
        ScriptBrowserEnv,
        StateInfo,
        Trajectory,
        action2str,
        create_id_based_action,
        create_stop_action,
    )
    from evaluation_harness.evaluators import evaluator_router

    # maintain a trajectory
    trajectory: Trajectory = []

    # Maintain element pools
    website_name = config["sites"][0]
    unclick_elem_pool_path = os.path.join(element_dir, website_name + "_unclick.json")
    click_elem_pool_path = os.path.join(element_dir, website_name + "_click.json")
    explored_elem_pool_path = os.path.join(element_dir, website_name + "_explored.json")

    unclick_elem_pool = load_element_pool(unclick_elem_pool_path)
    click_elem_pool = load_element_pool(click_elem_pool_path)
    explored_elem_pool = load_element_pool(explored_elem_pool_path)

    env = None

    try:
        # Init the environment
        env = ScriptBrowserEnv(
            headless=False,
            slow_mo=100,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720},
        )

        # set the environment for the current example (website）
        env.reset(options={"config_file": config_randomwalk_path})
        obs, info = get_state(env)
        state_info: StateInfo = {"observation": obs, "info": info}  # Save each interface state using StateInfo
        trajectory.append(state_info)   # Record initial interface

        # Record elements from previous interface for comparing newly appeared elements
        prev_state_elements = set()

        # random walk num_step
        num_step = 5

        for i in range(num_step):

            # Get current interface state
            obs_before, info_before = get_state(env)
            actree_obs = obs_before["text"]
            print(actree_obs)

            # Get candidate interactive elements from current interface
            state_elements = list(info_before['observation_metadata']['text']['obs_nodes_info'].items())

            # Extract elements from current interface
            current_state_elements = set(extract_state_ele(info_before['observation_metadata']['text']['obs_nodes_info']))
            # Find newly appeared elements
            new_elements = current_state_elements - prev_state_elements

            # Select element
            selected_elem_info = select_element(
                state_elements,
                actree_obs,
                unclick_elem_pool,
                new_elements,
                explored_elem_pool
            )

            # If no selectable elements, skip this iteration
            if selected_elem_info is None:
                print("No clickable elements found.")
                break

            action_element_id, action_element, element_key = selected_elem_info

            # Use GPT to determine if selected element is inputable and choose corresponding action (click/type)
            gpt_response = generate_text_input(actree_obs, action_element['text'])
            if gpt_response.get('is_input'):
                type_content = gpt_response.get('input_content', 'Test Input')
                next_action_str = f"type [{action_element_id}] [{type_content}]"
                next_action = create_id_based_action(next_action_str)
            else:
                next_action_str = f"click [{action_element_id}]"
                next_action = create_id_based_action(next_action_str)

            print(f"Step {i}: {next_action_str}")

            # Execute action
            env.step(next_action)
            time.sleep(SLEEP)

            # Add element to explored set
            explored_elem_pool.add(element_key)

            # Get execution result
            obs_after, info_after = get_state(env)
            actree_obs = obs_after["text"]

            # Determine if action caused interface change through screenshot comparison
            if are_screen_identical(obs_before['image'], obs_after['image']):
                # If interface unchanged, add element to Unavailable Element Pool
                print("The pages are identical. Added to unclick_elem_pool.")
                if element_key not in click_elem_pool:   # If not in clickable set
                    unclick_elem_pool.add(element_key)
            else:
                # Interface changed, record action and subsequent interface in trajectory
                print("The pages have differences.")
                click_elem_pool.add(element_key)
                trajectory.append((next_action_str, action_element_id, action_element))
                state_info = {"observation": obs_after, "info": info_after}
                trajectory.append(state_info)

            # Update previous interface element set, regardless of whether page changed
            prev_state_elements = current_state_elements

    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(1)
        env.close()
        time.sleep(1)
    finally:
        if env is not None:
            env.close()
            time.sleep(1)

    # Record element pools
    save_element_pool(unclick_elem_pool, unclick_elem_pool_path)
    save_element_pool(click_elem_pool, click_elem_pool_path)
    save_element_pool(explored_elem_pool, explored_elem_pool_path)

    # Save trajectory data
    save_trajectory(unique_id_traj, trajectory, traj_dir, screen_dir, website_name, random_url)

    return trajectory


num_episode = 50
config_file = "config_files/60.json"    # Choose the corresponding config file based on the website to be sampled
config_randomwalk = "/Users/cckevin/Desktop/config_randomwalk"
traj_dir = "/Users/cckevin/Desktop/traj_results"
screen_dir = "/Users/cckevin/Desktop/screen_results"
element_dir = "/Users/cckevin/Desktop/click_elements"

for i in tqdm(range(num_episode)):
    print(f"Episode {i} random walk")
    try:
        random_walk_episode(config_file, traj_dir, screen_dir, element_dir, config_randomwalk)
    except Exception as e:
        print(f"An error occurred: {e}")
