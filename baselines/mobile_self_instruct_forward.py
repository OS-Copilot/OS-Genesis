import os
import random
import json
from openai import OpenAI

client = OpenAI(api_key='your-api-key-here')  


def generate_instruction_and_steps(few_shot_examples):
    examples_text = "\n\n".join(
        f"Instruction {i+1}: {example['instruction']}\nSteps:\n" +
        "\n".join(f"- {step}" for step in example['steps'])
        for i, example in enumerate(few_shot_examples)
    )
    prompt = (
        f"Based on the following examples, generate a new instruction that is only one sentence. "
        f"Then, plan out 1-5 steps needed to complete this task using common GUI operations like 'click...', 'input...', 'scroll...'. "
        f"Please provide the instruction and the steps in the same format.\n\nExamples:\n\n{examples_text}\n\n"
        f"Please provide your instruction and steps below:"
    )

    messages = [
        {"role": "system", "content": "You are an AI assistant that generates instructions for computer agents based on input interface screenshots."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(model="gpt-4o", 
        messages=messages,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7)
    except Exception as e:
        print(f"Error generating instruction: {e}")
        return None, None

    content = response.choices[0].message.content.strip()

    instruction = ''
    steps = []
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Instruction'):
            instruction = line.partition(':')[2].strip()
            i += 1
            continue
        if line.startswith('Steps'):
            i += 1 
            while i < len(lines) and (lines[i].startswith('-') or lines[i].strip() == ''):
                if lines[i].startswith('-'):
                    steps.append(lines[i].lstrip('- ').strip())
                i += 1
            break
        i += 1

    return instruction, steps

def main():
    # 读取之前生成的指令
    json_filename = 'instructions.json'

    if not os.path.exists(json_filename):
        print(f"File {json_filename} does not exist.")
        return

    with open(json_filename, 'r', encoding='utf-8') as f:
        try:
            instructions_list = json.load(f)
        except json.JSONDecodeError:
            print("Read instructions.json error")
            return

    if len(instructions_list) < 3:
        print("instructions.json sample error")
        return

    num_instructions_to_generate = 1000
    new_instructions_list = []

    for i in range(num_instructions_to_generate):
        few_shot_examples = random.sample(instructions_list, 3)

        instruction, steps = generate_instruction_and_steps(few_shot_examples)
        if instruction is None or steps is None:
            continue  

        print(f"Progress: {i+1}/{num_instructions_to_generate}")
        print(f"Instruction: {instruction}")
        print("Steps:")
        for step in steps:
            print(f"- {step}")
        print("\n")

        instruction_dict = {
            "episode_id": "self-instruct",
            "step_id": 1,
            "app_name": "default",
            "task_name": "default",
            "instruction": instruction,
            "steps": steps
        }

        new_instructions_list.append(instruction_dict)

        with open('self_ins_instructions.json', 'w', encoding='utf-8') as f:
            json.dump(new_instructions_list, f, indent=4, ensure_ascii=False)

    print("Done in self_ins_instructions.json。")

if __name__ == "__main__":
    main()
