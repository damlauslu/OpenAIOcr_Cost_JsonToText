import os
import json

## NOT IN USE ANYMORE ##
def extract_all_text_from_json(data, depth=0):
    """Recursively extracts all text-like content from JSON data."""
    text_lines = []
    indent = "    " * depth  # For better readability in output

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

def extract_text_from_json(json_path, output_dir):
    """Extracts all text content from a JSON file and writes it to a .txt file."""
    try:
        # Check if JSON file exists
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Extract all text recursively
        extracted_text = "\n".join(extract_all_text_from_json(data))

        # Get file name without extension
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_subdir = os.path.join(output_dir, base_name)
        output_path = os.path.join(output_subdir, f"{base_name}.txt")

        # Ensure the output directory exists
        os.makedirs(output_subdir, exist_ok=True)

        # Write text to the output file
        with open(output_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(extracted_text)

        print(f"Text successfully extracted to {output_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def process_all_json_files(input_dir, output_dir):
    """Processes all JSON files in the input directory."""
    try:
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        # Iterate through all JSON files in the directory
        for file_name in os.listdir(input_dir):
            if file_name.lower().endswith('.json'):
                json_path = os.path.join(input_dir, file_name)
                extract_text_from_json(json_path, output_dir)

    except NotADirectoryError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Example usage
    json_directory = "./Data/"  # Directory containing JSON files
    results_directory = "./Results/"  # Directory to save the output text files

    process_all_json_files(json_directory, results_directory)
