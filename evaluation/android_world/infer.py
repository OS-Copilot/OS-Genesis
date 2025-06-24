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

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests
# from lmdeploy.serve.openai.api_client import APIClient
import base64
import os
import re
from openai import OpenAI
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# api_client = APIClient('http://101.42.242.98:1119')
# model_name = api_client.available_models[0]


ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini GCP interface."""

  def __init__(
      self,
      model_name: str | None = None,
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    if 'GCP_API_KEY' not in os.environ:
      raise RuntimeError('GCP API key not set.')
    genai.configure(api_key=os.environ['GCP_API_KEY'])
    self.llm = genai.GenerativeModel(
        model_name,
        safety_settings=None
        if enable_safety_checks
        else SAFETY_SETTINGS_BLOCK_NONE,
        generation_config=generation_types.GenerationConfig(
            temperature=temperature, top_p=top_p, max_output_tokens=1000
        ),
    )
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)

  def predict(
      self,
      text_prompt: str,
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(
        text_prompt, [], enable_safety_checks, generation_config
    )

  def is_safe(self, raw_response):
    try:
      return (
          raw_response.candidates[0].finish_reason
          != answer_types.FinishReason.SAFETY
      )
    except Exception:  # pylint: disable=broad-exception-caught
      #  Assume safe if the response is None or doesn't have candidates.
      return True

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    counter = self.max_retry
    retry_delay = 1.0
    output = None
    while counter > 0:
      try:
        output = self.llm.generate_content(
            [text_prompt] + [Image.fromarray(image) for image in images],
            safety_settings=None
            if enable_safety_checks
            else SAFETY_SETTINGS_BLOCK_NONE,
            generation_config=generation_config,
        )
        return output.text, True, output
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2

    if (output is not None) and (not self.is_safe(output)):
      return ERROR_CALLING_LLM, False, output
    return ERROR_CALLING_LLM, None, None

  def generate(
      self,
      contents: (
          content_types.ContentsType | list[str | np.ndarray | Image.Image]
      ),
      safety_settings: safety_types.SafetySettingOptions | None = None,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Any]:
    """Exposes the generate_content API.

    Args:
      contents: The input to the LLM.
      safety_settings: Safety settings.
      generation_config: Generation config.

    Returns:
      The output text and the raw response.
    Raises:
      RuntimeError:
    """
    counter = self.max_retry
    retry_delay = 1.0
    response = None
    if isinstance(contents, list):
      contents = self.convert_content(contents)
    while counter > 0:
      try:
        response = self.llm.generate_content(
            contents=contents,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        return response.text, response
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2
    raise RuntimeError(f'Error calling LLM. {response}.')

  def convert_content(
      self,
      contents: list[str | np.ndarray | Image.Image],
  ) -> content_types.ContentsType:
    """Converts a list of contents to a ContentsType."""
    converted = []
    for item in contents:
      if isinstance(item, str):
        converted.append(item)
      elif isinstance(item, np.ndarray):
        converted.append(Image.fromarray(item))
      elif isinstance(item, Image.Image):
        converted.append(item)
    return converted


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """

  # self.model = '/cpfs01/user/wuzhenyu/LLMs/InternVL2-4B'    
  # self.url = "http://101.42.242.98:1118/v1/chat/completions"

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      max_retry: int = 1,
      temperature: float = 0.0,
  ):
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 1
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    # self.model = '/cpfs01/user/wuzhenyu/LLMs/InternVL2-8B' 
    # self.model = 'internvl2-8b-sim'
    # self.model = 'Intern4b-baseline'

    ## 轨迹训练 4b
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_12k_high_traj_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    ## 轨迹训练 4b

    # ## 轨迹训练 8b
    # self.model = 'internvl2-8b-sim'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## 轨迹训练 8b

    # ## Hybrid训练 8b
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_22k_hybrid_zs3_bs16_8gpus'
    # # print(self.model)
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## Hybrid训练 8b


    # ## Hybrid训练 26b
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_26b_aw_stage1_gpt_high_level_aw_explore_22k_hybrid_zs3_bs32_32gpus'
    # # print(self.model)
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## Hybrid训练 26b

    # ## Hybrid训练 带 rm 4b v1.1
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_22k_hybrid_rm_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:1119/v1/chat/completions"
    # ## Hybrid训练 带 rm 4b v1.1 run_20241127T194218498127 还没测完

    # ## Hybrid训练 带 rm 8b v1.1
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_22k_hybrid_rm_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## Hybrid训练 带 rm 8b v1.1 run_20241127T194218498127


    # ## qwen 7b baseline
    # self.model = '/nas/shared/NLP_A100/wuzhenyu/LLMs/Qwen2-VL-7B-Instruct'
    # self.url = "http://101.42.242.98:1119/v1/chat/completions"
    # ## qwen 7b baseline

    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_12k_inst_action_traj_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"

    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_12k_inst_action_traj_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"


    # ## qwen 7b baseline fw
    # self.model = '/nas/shared/NLP_A100/wuzhenyu/ckpt/qwen2vl_7b_1128_sim_aw_high'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## qwen 7b baseline fw


    # ## Intern 8b forward + selfins
    # self.model = 'Intern8b-sim-finetuned'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## Intern 8b forward + selfins
    # # /nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_16k_self_inst_action_traj_zs3_bs16_8gpus

    # ## Intern 4b forward + selfins
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_16k_self_inst_action_traj_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## Intern 4b forward + selfins
    # # /nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_16k_self_inst_action_traj_zs3_bs16_8gpus

    # ## qwen7b forward + selfins
    # self.model = '/nas/shared/NLP_A100/wuzhenyu/ckpt/qwen2vl_7b_sim_16k_1202'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## qwen7b forward + selfins
    # # /nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_16k_self_inst_action_traj_zs3_bs16_8gpus

    # ## intern8b genesis v2 no answer
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_51k_v2_hybrid_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## intern8b genesis v2 no answer
    # # /nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_16k_self_inst_action_traj_zs3_bs16_8gpus

    # ## intern8b genesis v2.1
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_26k_v2_hybrid_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## intern8b genesis v2.1
    # # /nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_16k_self_inst_action_traj_zs3_bs16_8gpus


    # ## qwen7b genesis v2.1
    # self.model = '/nas/shared/NLP_A100/wuzhenyu/ckpt/qwen2vl_7b_sim_26k_1204'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"
    # ## qwen7b genesis v2.1
    # # /nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_16k_self_inst_action_traj_zs3_bs16_8gpus


    # # analysis no reward model all data
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_27k_v2_1_all_hybrid_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:1118/v1/chat/completions"

    # # analysis reward = 5
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_19k_v2_1_score5_hybrid_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:1118/v1/chat/completions"

    # # qwen 7 b全量消融
    # self.model = '/nas/shared/NLP_A100/wuzhenyu/ckpt/qwen2vl_7b_sim_27k_1208'
    # self.url = "http://101.42.242.98:1119/v1/chat/completions"

    # # qwen 7 b rm=5 消融
    # self.model = '/nas/shared/NLP_A100/wuzhenyu/ckpt/qwen2vl_7b_sim_19k_1208'
    # self.url = "http://101.42.242.98:1119/v1/chat/completions"

    # # intern8b scale=200 
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_200_traj_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"

    # # intern8b scale=500 
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_500_traj_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:1118/v1/chat/completions"

    # # intern4b scale=500 
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_500_traj_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:1118/v1/chat/completions"

    # # intern8b scale=1500 
    # self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_8b_aw_stage1_gpt_high_level_aw_explore_1500_traj_zs3_bs16_8gpus'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"

    # # qwen7b scale=1500 
    # self.model = '/nas/shared/NLP_A100/wuzhenyu/ckpt/qwen2vl_7b_sim_1223_aw_1500_traj'
    # self.url = "http://101.42.242.98:8888/v1/chat/completions"

    # intern4b scale=1500 
    self.model = '/nas/shared/NLP_A100/dingzichen/models/OS-Sim/finetune_internvl2_4b_aw_stage1_gpt_high_level_aw_explore_1500_traj_zs3_bs16_8gpus'
    self.url = "http://101.42.242.98:1118/v1/chat/completions"

    # self.model = '/nas/shared/NLP_A100/wuzhenyu/LLMs/InternVL2-8B'
    # self.url = "http://101.42.242.98:1119/v1/chat/completions"
  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openai_api_key}',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': 200,
        "stop": ["<|end|>", "<|im_end|>", "<|endoftext|>", "\n"]
    }

    # Gpt-4v supports multiple images, just need to insert them in the content
    # list.
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

          # print pay load
    # print('\n=== Payload Content ===')
    # print(f"Model: {payload['model']}")
    # print(f"Temperature: {payload['temperature']}")
    # print(f"Max tokens: {payload['max_tokens']}")
    # print('\nMessages:')
    # for message in payload['messages']:
    #     print(f"\nRole: {message['role']}")
    #     for content in message['content']:
    #         if content['type'] == 'text':
    #             print(f"Text: {content['text']}")
    #         elif content['type'] == 'image_url':
    #             print(f"Image data preview: {content['image_url']['url'][:100]}...")
    # print('=== End of Payload ===\n')

    counter = 1
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            self.url,
            headers=headers,
            json=payload,
        )
        # print('LLM Response:', response.json())
        if response.ok and 'choices' in response.json():
          # print(response.json()['choices'][0]['message']['content'])
          return (
              response.json()['choices'][0]['message']['content'],
              None,
              response,
          )
        print(
            'Error calling OpenAI API with error message: '
            + response.json()['error']['message']
        )
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None