import base64
import json
import os
from datetime import datetime
from openai import OpenAI

def encode_image(image_path):
    """Encodes an image to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error encoding image: {e}")

def process_response_to_txt(response, output_dir, image_local):
    """Processes the response from the API and saves it as a TXT file."""
    try:
        json_string = response.choices[0].message.content
        # Clean the raw response
        json_string = json_string.strip("```").strip()
        print("Raw response from API:", json_string)  # Display raw response for debugging
        json_string = json_string.replace("json\n", "").replace("\n", "")
        json_data = json.loads(json_string)

        # Extract all text recursively
        def extract_all_text_from_json(data, depth=0):
            text_lines = []
            indent = "    " * depth
            if isinstance(data, dict):
                for key, value in data.items():
                    text_lines.append(f"{indent}{key}:")
                    text_lines.extend(extract_all_text_from_json(value, depth + 1))
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    text_lines.append(f"{indent}- Item {idx + 1}:")
                    text_lines.extend(extract_all_text_from_json(item, depth + 1))
            elif isinstance(data, str):
                text_lines.append(f"{indent}{data}")
            elif isinstance(data, (int, float)):
                text_lines.append(f"{indent}{data}")
            return text_lines

        extracted_text = "\n".join(extract_all_text_from_json(json_data))

        filename_without_extension = os.path.splitext(os.path.basename(image_local))[0]
        txt_filename = f"{filename_without_extension}.txt"

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, txt_filename)

        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(extracted_text)

        print(f"Text data saved to {output_path}")
    except KeyError:
        raise ValueError("Unexpected response format: Missing 'choices' or 'message'.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from response: {e}")
    except Exception as e:
        raise RuntimeError(f"Error processing response: {e}")

def validate_image(image_path):
    """Validates if the image file exists and is of a supported format."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        raise ValueError(f"Unsupported image format: {image_path}")

def calculate_cost(total_tokens, model_name):
    """Calculates the API cost based on token usage and model."""
    cost_per_1000_tokens = {
        "gpt-3.5-turbo": 0.0015,  # Input cost per 1000 tokens
        "gpt-4": 0.03,           # Input cost per 1000 tokens
        "gpt-4o": 0.00015,       # Input cost per 1000 tokens (example GPT-4o input cost)
    }
    if model_name not in cost_per_1000_tokens:
        raise ValueError(f"Unknown model '{model_name}'. Cannot calculate cost.")

    return (total_tokens / 1000) * cost_per_1000_tokens[model_name]

def save_cost_to_file(total_cost, image_name, output_dir):
    """Appends the cost, image name, and timestamp to a txt file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        cost_filename = 'cost.txt'
        cost_output_path = os.path.join(output_dir, cost_filename)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(cost_output_path, 'a') as file:
            file.write(f"{timestamp} - Image: {image_name}, Estimated API cost: ${total_cost:.6f}\n")

        print(f"Cost data appended to {cost_output_path}")
    except Exception as e:
        print(f"Error saving cost data: {e}")

def main():
    """Main function to handle the workflow."""
    image_local = 'deneme_tr_images/turkce_elyazisi.png'
    txt_output_dir = './Results/'
    cost_output_dir = './costdata/'

    # Ensure output directories exist before API request
    os.makedirs(txt_output_dir, exist_ok=True)
    os.makedirs(cost_output_dir, exist_ok=True)
    print(f"Output directories are ready: {txt_output_dir}, {cost_output_dir}")

    # Validate image
    try:
        validate_image(image_local)
        print("Image validation passed.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Validation error: {e}")
        return

    # Encode image
    try:
        base64_img = f"data:image/png;base64,{encode_image(image_local)}"
    except RuntimeError as e:
        print(f"Image encoding error: {e}")
        return

    # Initialize OpenAI client
    api_key_used = False
    client = OpenAI(api_key="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")

    # Make API request
    try:
        model_name = 'gpt-4o'
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return the JSON document with the data. Ensure only JSON is returned, without any additional text. If Turkish characters exist, replace them with their English equivalents (e.g., ü -> u, ş -> s, ç -> c, ğ -> g, ı -> i, ö -> o) to prevent errors."},
                    {"type": "image_url", "image_url": {"url": f"{base64_img}"}}
                ],
            }],
            max_tokens=2000,
        )
        api_key_used = True

        # Calculate and display cost
        try:
            total_tokens = response.usage.total_tokens  # Correct attribute access
            total_cost = calculate_cost(total_tokens, model_name)
            print(f"Total tokens used: {total_tokens}")
            print(f"Estimated API cost: ${total_cost:.6f}")

            # Save cost data to file in the 'costdata' directory
            save_cost_to_file(total_cost, os.path.basename(image_local), cost_output_dir)

        except AttributeError:
            print("Token usage information not available in response.")

        # Save extracted text from JSON response to TXT file in the 'Results' directory
        process_response_to_txt(response, txt_output_dir, image_local)

    except Exception as e:
        print(f"API request or processing error: {e}")
    finally:
        if api_key_used:
            print("API key was used in this operation.")

if __name__ == "__main__":
    main()
