import requests
import json
import os
import dashscope


dashscope.api_key = "Your Qwen API"

def send_request(messages, model='qwen-plus'):
    """发送POST请求到指定的API, 并返回响应数据。"""
    print("Querying Qwen LLM ...")
    response = dashscope.Generation.call(model='qwen-plus', messages=messages)

    if response.status_code == 200:
        return response
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None

def generate_plan(prompt_text):
    """
    调用Qwen 模型 API 来生成文本。

    参数:
    - prompt_text (str): 用于生成文本的 prompt。

    返回:
    - 生成的文本 (str)，或错误信息。
    """
    
    print("Start generating questions ...")

    messages = prompt_text 


    response_data = send_request(messages, model='qwen-plus')
    response_dict = response_data

    content = response_dict['output']['text'] 
    #content =  response_dict
    return content if content else print('Failed to get response from Qwen-Plus')

def read_prompt(file_path):
    """从指定文件中读取prompt内容。"""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

def write_result(file_path, text):
    """将生成的文本写入指定文件."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'w') as file:
            file.write(text)
        print(f"Generated Text has been written to {file_path}")
    except IOError as e:
        print(f"Error writing to {file_path}: {e}")

def use_cached_plan(cache_file):
    """读取并打印缓存的生成计划。"""
    cached_text = read_prompt(cache_file)
    if cached_text:
        print("Cached Generated Text:")
        print(cached_text)
