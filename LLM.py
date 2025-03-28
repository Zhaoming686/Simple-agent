from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict]):
        pass

    def load_model(self):
        pass

# class InternLM2Chat(BaseModel):
class TinyLlamaChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def load_model(self):
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
        # 改成CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.float32,  # <-- 改成 float32, float16在CPU上不一定支持
            trust_remote_code=True
        ).to("cpu").eval()

        print('================ Model loaded ================')

    # def chat(self, prompt: str, history: List[dict], meta_instruction:str ='') -> str:
    #     response, history = self.model.chat(self.tokenizer, prompt, history, temperature=0.1, meta_instruction=meta_instruction)
    #     return response, history
    def chat(self, prompt: str, history: List[dict], meta_instruction: str = '') -> str:
        print("DEBUG: meta_instruction is", type(meta_instruction))
        print("DEBUG: prompt is", type(prompt))

        meta_instruction = meta_instruction or ""
        prompt = prompt or ""  # 添加这行，防止 prompt 是 None

        # 拼接 history（TinyLlama 不支持真实 history，我们模拟对话）
        if history:
            dialogue = ''.join([f"<user>{h['user']}\n<assistant>{h['assistant']}\n" for h in history])
        else:
            dialogue = ''
        full_prompt = meta_instruction + '\n' + dialogue + prompt

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cpu")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 模拟输出 history 格式
        history.append({"user": prompt, "assistant": response})
        return response, history
# if __name__ == '__main__':
#     model = InternLM2Chat('/root/share/model_repos/internlm2-chat-7b')
#     print(model.chat('Hello', []))