import os
import random
import json
import base64
from openai import OpenAI

client = OpenAI(api_key='your-api-key-here')  # 替换为你的 OpenAI API 密钥

def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_instruction_and_steps(few_shot_examples, screenshot_path):
    base64_image = encode_image_to_base64(screenshot_path)
    
    examples_text = "\n\n".join(
        f"Instruction {i+1}: {example['instruction']}\nSteps:\n" + "\n".join(f"- {step}" for step in example['steps'])
        for i, example in enumerate(few_shot_examples)
    )
    
    prompt = (
        f"Based on the following examples and the provided screenshot, generate a new instruction that is only one sentence. "
        f"The instruction should be related to what can be done in the interface shown in the screenshot. "
        f"Then, plan out 1-5 steps needed to complete this task using common GUI operations like 'click...', 'input...', 'scroll...', 'drag...'. "
        f"Please provide the instruction and the steps in the same format.\n\nExamples:\n\n{examples_text}\n\n"
        f"Please provide your instruction and steps below:"
    )

    messages = [
        {"role": "system", "content": "You are an AI assistant that generates instructions for computer agents based on input interface screenshots."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
    except Exception as e:
        print(f"Error generating instruction: {e}")
        return None, None

    content = response.choices[0].message.content.strip()

    instruction = ''
    steps = []
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('Instruction'):
            instruction = line.partition(':')[2].strip()
            i += 1
            continue
        if line.startswith('Steps'):
            i += 1
            while i < len(lines) and lines[i].startswith('-'):
                steps.append(lines[i].lstrip('- ').strip())
                i += 1
            break
        i += 1

    return instruction, steps

def main():
    screenshot_dir = 'init_screenshots'
    screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]

    if not screenshots:
        print("No screenshots found in the directory.")
        return

    # Few-shot 示例
    few_shot_examples = [
        {
            "instruction": "Add a new contact named 'Alice Smith' with phone number '123-456-7890' in the contacts app.",
            "steps": [
                "Click the 'Contacts' app to open it.",
                "Tap the 'Add New Contact' button.",
                "Input 'Alice Smith' into the 'Name' field.",
                "Input '123-456-7890' into the 'Phone Number' field.",
                "Tap 'Save' to store the new contact."
            ]
        },
        {
            "instruction": "Set an alarm for 7:00 AM tomorrow using the clock app.",
            "steps": [
                "Open the 'Clock' app by clicking its icon.",
                "Navigate to the 'Alarm' tab.",
                "Click the '+' button to add a new alarm.",
                "Set the time to '7:00 AM' for tomorrow.",
                "Tap 'Save' to activate the alarm."
            ]
        },
        {
            "instruction": "Send an email to 'bob@example.com' with the subject 'Meeting Update' and attach the latest report.",
            "steps": [
                "Open the 'Email' app.",
                "Tap the 'Compose' button to start a new email.",
                "Input 'bob@example.com' into the 'To' field.",
                "Input 'Meeting Update' into the 'Subject' field.",
                "Tap the 'Attach' button and select the latest report file.",
                "Tap 'Send' to send the email."
            ]
        }
    ]

    instructions_list = []
    json_filename = 'mobile_instructions_forward.json'

    if os.path.exists(json_filename):
        with open(json_filename, 'r', encoding='utf-8') as f:
            try:
                instructions_list = json.load(f)
            except json.JSONDecodeError:
                print("Error decoding JSON from existing instructions.json file. Starting with an empty list.")
                instructions_list = []

    num_instructions = 1000

    for i in range(num_instructions):
        screenshot = random.choice(screenshots)
        episode_id = screenshot.replace('.png', '')
        screenshot_path = os.path.join(screenshot_dir, screenshot)

        instruction, steps = generate_instruction_and_steps(few_shot_examples, screenshot_path)
        if instruction is None or steps is None:
            continue

        print(f"Progress: {i+1}/{num_instructions}")
        print(f"Episode ID: {episode_id}")
        print(f"Instruction: {instruction}")
        print("Steps:")
        for step in steps:
            print(f"- {step}")
        print("\n")

        instruction_dict = {
            "episode_id": episode_id,
            "step_id": 1,
            "app_name": "default",
            "task_name": "default",
            "instruction": instruction,
            "steps": steps
        }

        instructions_list.append(instruction_dict)

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(instructions_list, f, indent=4, ensure_ascii=False)

    print("All instructions have been generated and saved to instructions.json.")

if __name__ == "__main__":
    main()