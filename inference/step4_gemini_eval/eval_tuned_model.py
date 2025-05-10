import os
import argparse
from datetime import datetime
import time
import signal
from PIL import Image
from google import genai
from google.genai import types
import pandas as pd

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Request timed out")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('-o', '--output_file', type=str, default="test_results.txt", 
                        help='Path to save the results')
    args = parser.parse_args()

    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', 1000)
    df = pd.read_csv('./text2cad_v1.1.csv')
    
    image_dir = args.image_dir
    result_output_file = args.output_file
    results = []
    failed_cases = []
    
    # Initialize Gemini API client
    client = genai.Client(api_key="")
    
    # Set constants
    timeout = 60
    max_retries = 3
    retry_delay = 10
    
    for image_file in sorted(os.listdir(image_dir)):
        if image_file.endswith('.png') and not image_file.startswith('._'):
            try:
                image_path = os.path.join(image_dir, image_file)
                image_index = image_file.split('.')[0]
                image = Image.open(image_path)
                row = df[df['uid'] == f"{image_index[:4]}/{image_index}"].iloc[0]
                
                # Set timeout for API call
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        config=types.GenerateContentConfig(
                            temperature=0,
                            system_instruction='''
                            Format your response like this:
                            Match: Yes or No  
                            Do **not** provide any extra explanation or commentary beyond this format.
                            Only return the response in the exact format above.  
                            '''
                        ),
                        contents=[image, f'''
                    You are a product design engineer.
                    Evaluate the 3D CAD model shown in the image using the following description:  
                    "{row['description']}"
                    1. Does the model **roughly match** this description?  
                    Please answer only `Yes` or `No`.  
                    Note: Since the image is a single-angle rendering, some features may not be visible.  
                    If it is **reasonably possible** that the model matches the description, please answer `Yes`.  
                    Only respond with `No` if you are **very certain** that the model does not match.       
                    ''']
                    )
                    print(f"{image_index}: {response.text}")
                    results.append(f"{image_index}: {response.text}")
                except (TimeoutException, Exception) as e:
                    print(f"Processing failed for {image_index}: {e}")
                    failed_cases.append((image_path, image_index, row))
                finally:
                    signal.alarm(0)  # Disable alarm
            
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
    
    # Retry failed cases
    if failed_cases:
        print(f"\nRetrying {len(failed_cases)} failed cases...")
        
        for retry_count in range(max_retries):
            if not failed_cases:
                break
                
            print(f"\nRetry attempt {retry_count + 1}/{max_retries}")
            still_failed = []
            
            for image_path, image_index, row in failed_cases:
                print(f"Retrying {image_index}...")
                try:
                    # Wait before retrying to avoid rate limits
                    time.sleep(retry_delay)
                    
                    image = Image.open(image_path)
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout)
                    
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            config=types.GenerateContentConfig(
                                temperature=0,
                                system_instruction='''
                                Format your response like this:
                                Match: Yes or No  
                                Do **not** provide any extra explanation or commentary beyond this format.
                                Only return the response in the exact format above.  
                                '''
                            ),
                            contents=[image, f'''
                            You are a product design engineer.
                            Evaluate the 3D CAD model shown in the image using the following description:  
                            "{row['description']}"
                            1. Does the model **roughly match** this description?  
                            Please answer only `Yes` or `No`.  
                            Note: Since the image is a single-angle rendering, some features may not be visible.  
                            If it is **reasonably possible** that the model matches the description, please answer `Yes`.  
                            Only respond with `No` if you are **very certain** that the model does not match.       
                            ''']
                        )
                        print(f"{image_index}: {response.text}")
                        results.append(f"{image_index}: {response.text}")
                    except (TimeoutException, Exception) as e:
                        print(f"Retry failed for {image_index}: {e}")
                        still_failed.append((image_path, image_index, row))
                    finally:
                        signal.alarm(0)  # Disable alarm
                except Exception as e:
                    print(f"Error retrying {image_index}: {e}")
                    still_failed.append((image_path, image_index, row))
            
            failed_cases = still_failed
            print(f"Remaining failed cases: {len(failed_cases)}")
    
    # Report any cases that still failed after all retries
    if failed_cases:
        print(f"\nWARNING: {len(failed_cases)} cases could not be processed after all retries:")
        for _, image_index, _ in failed_cases:
            print(f"  - {image_index}")

    # Save results to file
    try:
        with open(result_output_file, "w") as f:
            for line in results:
                f.write(line + "\n")
        print(f"\nResults saved to {result_output_file}")
    except Exception as e:
        print(f"Error writing results to file: {e}")

if __name__ == "__main__":
    start = datetime.now()
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        end = datetime.now()
        print(f"Start: {start.strftime('%H:%M:%S')}")
        print(f"End:   {end.strftime('%H:%M:%S')}")
        print(f"Total execution time: {end - start}")