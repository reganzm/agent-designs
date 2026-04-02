"""
complete_prompt_chain.py - 完整的规格提取与转换系统
包含错误处理、批量处理、验证等完整功能
"""

import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 优先加载脚本所在目录的 .env，避免因运行目录不同而读不到
_load_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_load_env)
load_dotenv()  # 再加载当前工作目录的 .env

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError(
        "未设置 DEEPSEEK_API_KEY。请在项目根目录创建 .env 文件并写入：\n"
        "  DEEPSEEK_API_KEY=你的API密钥\n"
        "或设置系统环境变量。API Key 申请：https://platform.deepseek.com/api_keys"
    )


class SpecificationExtractor:
    """规格提取器类"""
    
    def __init__(self):
        self.llm = ChatDeepSeek(
            model="deepseek-reasoner",
            temperature=0,
            max_tokens=None,
        )
        self._setup_chains()
    
    def _setup_chains(self):
        """设置提示链"""
        # 提取链
        prompt_extract = ChatPromptTemplate.from_template(
            "Extract the technical specifications from the following text:\n\n{text_input}"
        )
        self.extraction_chain = prompt_extract | self.llm | StrOutputParser()
        
        # 完整链
        prompt_transform = ChatPromptTemplate.from_template(
            "Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
        )
        self.full_chain = (
            {"specifications": self.extraction_chain}
            | prompt_transform
            | self.llm
            | StrOutputParser()
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract(self, text: str) -> Optional[Dict]:
        """提取规格信息"""
        try:
            result = self.full_chain.invoke({"text_input": text})
            return self._validate_output(result)
        except Exception as e:
            print(f"提取失败: {e}")
            return None
    
    def _validate_output(self, output: str) -> Optional[Dict]:
        """验证输出格式"""
        try:
            data = json.loads(output)
            required = ['cpu', 'memory', 'storage']
            if all(field in data for field in required):
                return data
        except:
            # 尝试修复常见格式问题
            try:
                fixed = output.replace("'", '"').strip()
                return json.loads(fixed)
            except:
                return None
        return None
    
    def batch_extract(self, texts: List[str]) -> List[Optional[Dict]]:
        """批量提取"""
        results = []
        for text in texts:
            results.append(self.extract(text))
        return results

# 使用示例
if __name__ == "__main__":
    extractor = SpecificationExtractor()
    
    # 单条提取
    sample = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."
    result = extractor.extract(sample)
    print("单条提取结果:", result)
    
    # 批量提取
    samples = [
        "手机规格配置：骁龙888处理器，8GB内存，128GB存储",
        "服务器规格配置：双路Intel Xeon Gold 6348，256GB DDR4，4TB NVMe SSD"
    ]
    batch_results = extractor.batch_extract(samples)
    print("批量提取结果:", batch_results)
