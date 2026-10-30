"""百炼接口与不含凭证的调用记录。"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


class BailianClient:
    def __init__(self, root: Path):
        load_dotenv(root / '.env', override=False)
        self.key = os.environ.get('DASHSCOPE_API_KEY', '')
        if not self.key:
            raise ValueError('请在 .env 中填写 DASHSCOPE_API_KEY')
        self.base_url = os.environ['QWEN_BASE_URL'].rstrip('/')
        self.dashscope_url = os.environ['DASHSCOPE_BASE_URL'].rstrip('/')
        self.model = os.environ.get('QWEN_MODEL', 'qwen3.8-27b')
        self.embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-v4')
        self.rerank_model = os.environ.get('RERANK_MODEL', 'qwen3-rerank')
        self.dimensions = int(os.environ.get('EMBEDDING_DIMENSIONS', '1024'))
        self.session = requests.Session()
        self.calls: list[dict] = []

    def _post(self, url: str, payload: dict, kind: str) -> dict:
        for attempt in range(3):
            started = time.perf_counter()
            try:
                response = self.session.post(url, json=payload,
                    headers={'Authorization': f'Bearer {self.key}'}, timeout=(20, 240))
            except requests.RequestException as error:
                self.calls.append({'kind': kind, 'model': payload['model'], 'ok': False,
                    'seconds': time.perf_counter() - started, 'error': type(error).__name__})
                if attempt == 2:
                    raise RuntimeError(f'{kind} 网络请求失败：{type(error).__name__}') from None
                time.sleep(2 ** attempt)
                continue
            record = {'kind': kind, 'model': payload['model'], 'http_status': response.status_code,
                      'seconds': time.perf_counter() - started, 'ok': response.ok}
            if not response.ok:
                self.calls.append(record)
                if (response.status_code == 429 or response.status_code >= 500) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                # 外部报错先移除密钥，避免网关回显凭证。
                message = response.text.replace(self.key, '[REDACTED]')[:600]
                raise RuntimeError(f'{kind} HTTP {response.status_code}: {message}')
            data = response.json()
            record['usage'] = data.get('usage', {})
            record['request_id'] = data.get('request_id', data.get('id'))
            self.calls.append(record)
            return data
        raise RuntimeError('请求未完成')

    def embed(self, texts: list[str], text_type: str = 'document') -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 10):
            data = self._post(self.base_url + '/embeddings', {
                'model': self.embedding_model, 'input': texts[start:start + 10],
                'dimensions': self.dimensions, 'encoding_format': 'float'}, 'embedding')
            rows = sorted(data['data'], key=lambda row: row['index'])
            if len(rows) != len(texts[start:start + 10]):
                raise ValueError('Embedding 返回数量与输入不符')
            vectors.extend(row['embedding'] for row in rows)
        return vectors

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        data = self._post(self.dashscope_url + '/services/rerank/text-rerank/text-rerank', {
            'model': self.rerank_model, 'input': {'query': query, 'documents': documents},
            'parameters': {'top_n': top_n, 'return_documents': False}}, 'rerank')
        return data['output']['results']

    def chat(self, messages: list[dict]) -> dict:
        return self._post(self.base_url + '/chat/completions', {
            'model': self.model, 'messages': messages, 'temperature': 0,
            'seed': 20261018, 'max_tokens': 5000, 'enable_thinking': False,
            'response_format': {'type': 'json_object'}}, 'generation')

    def describe(self) -> dict:
        return {'generation_model': self.model, 'embedding_model': self.embedding_model,
                'rerank_model': self.rerank_model, 'embedding_dimensions': self.dimensions,
                'base_url': self.base_url, 'temperature': 0, 'seed': 20261018,
                'max_tokens': 5000, 'enable_thinking': False}
