import os
import random
import json
import base64
from openai import OpenAI

client = OpenAI(api_key='your-api-key-here')  # Replace with your OpenAI API key

def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_instruction_and_steps(few_shot_examples, screenshot_path):
    base64_image = encode_image_to_base64(screenshot_path)
    
    examples_text = "\n\n".join(
        f"Instruction {i+1}: {example['instruction']}\nSteps:\n" + "\n".join(f"- {step}" for step in example['steps'])
        for i, example in enumerate(few_shot_examples)
    )
    
    # prompt = (
    #     f"Based on the following examples and the provided screenshot, generate a new instruction that is only one sentence. "
    #     f"The instruction should be related to what can be done in the interface shown in the screenshot. "
    #     f"Then, plan out 1-5 steps needed to complete this task using common GUI operations like 'click...', 'input...', 'scroll...', 'drag...'. "
    #     f"Please provide the instruction and the steps in the same format.\n\nExamples:\n\n{examples_text}\n\n"
    #     f"Please provide your instruction and steps below:"
    # )

    prompt = (
    f"Based on the following examples and the provided screenshot, generate a new instruction that is only one sentence. "
    f"The instruction should be related to what can be done in the interface shown in the screenshot. "
    f"Make sure to explore as many functionalities of the interface as possible while constructing the instruction. "
    f"Additionally, avoid clicking on 'starting map' or logging in for any map-related tasks.\n\n"
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
            max_tokens=500,
            temperature=1.1
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
    screenshot_dir = 'web_init_screenshots'
    screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]

    if not screenshots:
        print("No screenshots found in the directory.")
        return

    # Few-shot 示例
    few_shot_examples = [
        {
            "instruction": "Add a pair of Nike running shoes to your shopping cart.",
            "steps": [
                "Click the search bar and type 'Nike running shoes'",
                "Select your desired shoe size from the dropdown menu",
                "Click the 'Add to Cart' button"
            ]
        },
        {
            "instruction": "Search for a hotel in New York for next weekend and filter for 4-star hotels under $200 per night.",
            "steps": [
                "Enter 'New York' in the destination search box",
                "Select next weekend's dates in the calendar",
                "Click the 'Search' button",
                "Click 'Filter' or 'Refine Search'",
                "Select '4 stars' in the hotel class filter",
                "Set the price range maximum to $200",
                "Click 'Apply Filters' to see results"
            ]
        },
        {
            "instruction": "Apply a promotional discount code to your cart.",
            "steps": [
                "Click on the shopping cart icon",
                "Enter the promotion code in the discount field",
                "Click 'Apply' to validate the code"
            ]
        },
        {
        "instruction": "Create a new repository on GitLab.",
        "steps": [
            "Click on the 'New Project' button or menu option",
            "Select 'Create Blank Project' from the options",
            "Enter a repository name in the 'Project Name' field",
            "Set the visibility level (e.g., Public, Private)",
            "Click 'Create Project' to finalize"
        ]
        },
        {
        "instruction": "Create a new post in a subreddit of your choice.",
        "steps": [
            "Click on the 'Create Post' button or link",
            "Select the desired subreddit from the dropdown or search bar",
            "Enter the post title in the 'Title' field",
            "Write your post content in the text box provided",
            "Click 'Post' to publish"
        ]
    },
    ]

    instructions_list = []
    json_filename = 'web_instructions.json'

    if os.path.exists(json_filename):
        with open(json_filename, 'r', encoding='utf-8') as f:
            try:
                instructions_list = json.load(f)
            except json.JSONDecodeError:
                print("Error decoding JSON from existing instructions.json file. Starting with an empty list.")
                instructions_list = []

    num_instructions = 600

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


# few_shot_examples = [
#         {
#             "instruction": "Add a black Nike running shoes to your shopping cart on the e-commerce website.",
#             "steps": [
#                 "Click on the 'Shoes' category in the navigation menu",
#                 "Click the filter button and select 'Nike' under brand",
#                 "Scroll down to find the black running shoes",
#                 "Click on the desired Nike running shoes to view details",
#                 "Select your size from the size dropdown menu",
#                 "Click the 'Add to Cart' button"
#             ]
#         },
#         {
#             "instruction": "Search for a hotel in New York for next weekend and filter for 4-star hotels under $200 per night.",
#             "steps": [
#                 "Enter 'New York' in the destination search box",
#                 "Select next weekend's dates in the calendar",
#                 "Click the 'Search' button",
#                 "Click 'Filter' or 'Refine Search'",
#                 "Select '4 stars' in the hotel class filter",
#                 "Set the price range maximum to $200",
#                 "Click 'Apply Filters' to see results"
#             ]
#         },
#         {
#             "instruction": "Complete the purchase of items in your shopping cart using a credit card.",
#             "steps": [
#                 "Click the shopping cart icon",
#                 "Review the items in your cart",
#                 "Click 'Proceed to Checkout'",
#                 "Fill in shipping address details",
#                 "Select 'Credit Card' as payment method",
#                 "Enter credit card information",
#                 "Click 'Place Order' to complete purchase"
#             ]
#         },
#         {
#             "instruction": "Sign up for a new account on the e-commerce platform.",
#             "steps": [
#                 "Click the 'Sign Up' or 'Register' button",
#                 "Enter your email address in the email field",
#                 "Create and enter a password in the password field",
#                 "Fill in your personal information (name, phone number)",
#                 "Check the terms and conditions box",
#                 "Click the 'Create Account' button"
#             ]
#         },
#         {
#             "instruction": "Apply filters to find a laptop under $1000 with at least 16GB RAM.",
#             "steps": [
#                 "Click on the 'Electronics' or 'Computers' category",
#                 "Select 'Laptops' from the subcategories",
#                 "Click on 'Filter' options",
#                 "Set price range maximum to $1000",
#                 "Select '16GB or more' under RAM filter",
#                 "Click 'Apply Filters'",
#                 "Sort results by price low to high"
#             ]
#         },
#         {
#             "instruction": "Write and submit a product review for your recent purchase.",
#             "steps": [
#                 "Navigate to 'My Orders' or 'Order History'",
#                 "Find the product you want to review",
#                 "Click 'Write a Review' button",
#                 "Select star rating (1-5 stars)",
#                 "Write your review in the text box",
#                 "Upload product photos if desired",
#                 "Click 'Submit Review' button"
#             ]
#         },
#         {
#             "instruction": "Subscribe to the newsletter and apply a promotional discount code.",
#             "steps": [
#                 "Locate the newsletter subscription box",
#                 "Enter your email address",
#                 "Click 'Subscribe' button",
#                 "Find the promotion code entry field",
#                 "Input the promotional code",
#                 "Click 'Apply' to validate the code"
#             ]
#         },
#         {
#             "instruction": "Compare specifications of two different smartphone models.",
#             "steps": [
#                 "Search for the first smartphone model",
#                 "Click 'Add to Compare' on the first phone",
#                 "Search for the second smartphone model",
#                 "Click 'Add to Compare' on the second phone",
#                 "Click 'Compare' or 'View Comparison'",
#                 "Scroll through the comparison chart"
#             ]
#         }
#     ]