from google import genai
from google.genai import types
import os
import subprocess
import argparse
from prompts import PROMPT
from datetime import datetime
import time
import signal

client = genai.Client(api_key="")

def get_input(json_path, file_index, save_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_str = f.read()
    return f"'''Give me CAD query from this CAD sequence: {json_str}. The export file name should be {os.path.join(save_path, file_index)}.stl. In the end, only save stl file, don't need to use show().'''"


def get_pure_python(gemini_output):
    if gemini_output.startswith("```python"):
        gemini_output = gemini_output[len("```python"):].strip()
    if gemini_output.endswith("```"):
        gemini_output = gemini_output[:-len("```")].strip()
    return gemini_output


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Request timed out")


def generate_code_with_timeout(chat, user_input, timeout=60):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        response = chat.send_message(user_input)
        result = get_pure_python(response.text)
        signal.alarm(0)
        return result
    except TimeoutException:
        print(f"Generation timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error generating code: {e}")
        return None
    finally:
        signal.alarm(0)


def generate_and_validate_code(chat, json_str, file_index, cq_dir):
    response = generate_code_with_timeout(chat, json_str)
    if response is None:
        return False, os.path.join(cq_dir, f"{file_index}.py")
    cq_path = os.path.join(cq_dir, f"{file_index}.py")

    for attempt in range(2):
        with open(cq_path, "w", encoding="utf-8") as f:
            f.write(response)

        result = subprocess.run(['python', cq_path], capture_output=True, text=True)
        if result.returncode == 0:
            return True, cq_path
        else:
            if attempt == 0:
                print(f"{file_index} meets error")
                error_msg = '\n'.join(result.stderr.splitlines()[-5:])
                retry_prompt = f'code: {response} has an error: {error_msg}, generate it again, only give me python code'
                response = generate_code_with_timeout(chat, retry_prompt)
                if response is None:
                    return False, cq_path
    return False, cq_path


def main():
    success_count = 0
    fail_count = 0
    failed_scripts = []

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', type=str, required=True)
    parser.add_argument('-s', '--stl_dir', type=str, required=True)
    parser.add_argument('-c', '--cq_dir', type=str, required=True)
    args = parser.parse_args()
    root_dir = args.root_dir
    stl_dir = args.stl_dir
    cq_dir = args.cq_dir
    
    os.makedirs("./failed_scripts", exist_ok=True)
    
    for file in os.listdir(root_dir):
        if file.endswith('json'):
            print(f'working with {file}')
            file_path = os.path.join(root_dir, file)
            file_index = file.split('_')[0]
            json_str = get_input(file_path, file_index, stl_dir)
            chat = client.chats.create(model="gemini-2.0-flash", config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=PROMPT
            ))

            success, path = generate_and_validate_code(chat, json_str, file_index, cq_dir)
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_scripts.append(path)

    print(f"Number of success: {success_count}")
    print(f"Number of failure: {fail_count}")
    with open(os.path.join("./failed_scripts/", f'{os.path.basename(cq_dir)}_failed_scripts.txt'), 'w') as f:
        for item in failed_scripts:
            f.write(str(item) + '\n')


if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print(f"Start: {start.strftime('%H:%M:%S')}")
    print(f"End:   {end.strftime('%H:%M:%S')}")
    print(f"Total execution time: {end - start}")