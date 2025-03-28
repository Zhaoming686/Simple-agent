from typing import Dict, List, Optional, Tuple, Union
import json5

# from SimpleAgent.LLM import InternLM2Chat
from SimpleAgent.LLM import TinyLlamaChat
from SimpleAgent.tool import Tools


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """
You are an intelligent agent designed to answer user questions by interacting ONLY with tools provided to you.

You MUST follow these instructions carefully:

1. NEVER make up answers on your own.
2. ONLY use the tools listed below to find the answer.
3. Use the exact format specified.
4. Format all action input as a valid JSON object.
5. DO NOT continue after giving the Final Answer.

You have access to the following tool(s):

{tool_descs}

Use the following format **exactly**:

Question: the input question you must answer  
Thought: think step by step about what to do  
Action: the action to take, must be one of [{tool_names}]  
Action Input: the input to the action, must be a valid JSON object  
Observation: the result returned by the tool  
... (Repeat Thought/Action/Action Input/Observation if needed)  
Thought: I now know the final answer  
Final Answer: the final answer to the original question

---

Example:

Question: 今天的科技新闻有哪些？  
Thought: I need to search for today's tech news using google_search.  
Action: google_search  
Action Input: {{ "search_query": "今天的科技新闻" }}  
Observation: 今天谷歌发布了一个新AI模型，用于图像生成和理解。  
Thought: I now know the final answer.  
Final Answer: 今天谷歌发布了一个新AI模型，用于图像生成和理解。

---

Now begin!
"""

class Agent:
    def __init__(self, path: str = '') -> None:
        self.path = path
        self.tool = Tools()
        self.system_prompt = self.build_system_input()
        # self.model = InternLM2Chat(path)
        self.model = TinyLlamaChat(path)
    def build_system_input(self):
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt
    
    def parse_latest_plugin_call(self, text):
        plugin_name, plugin_args = '', ''
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
            plugin_name = text[i + len('\nAction:') : j].strip()
            plugin_args = text[j + len('\nAction Input:') : k].strip()
            text = text[:k]
        return plugin_name, plugin_args, text
    
    # def call_plugin(self, plugin_name, plugin_args):
    #     plugin_args = json5.loads(plugin_args)
    #     if plugin_name == 'google_search':
    #         return '\nObservation:' + self.tool.google_search(**plugin_args)
    def call_plugin(self, plugin_name, plugin_args):
        plugin_args = json5.loads(plugin_args)
        if plugin_name == 'google_search':
            return '\nObservation:' + self.tool.google_search(**plugin_args)
        else:
            return '\nObservation: [Error] Unsupported plugin: ' + plugin_name

    def text_completion(self, text, history=[]):
        text = "\nQuestion:" + text
        response, his = self.model.chat(text, history, self.system_prompt)
        print(response)
        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)
        if plugin_name:
            response += self.call_plugin(plugin_name, plugin_args)
        response, his = self.model.chat(response, history, self.system_prompt)
        return response, his

if __name__ == '__main__':
    agent = Agent("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    prompt = agent.build_system_input()
    print(prompt)