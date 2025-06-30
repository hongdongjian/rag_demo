import unittest
from openai import OpenAI


class MyTestCase(unittest.TestCase):
    def test_something(self):
        import base64
        import numpy as np

        # 向量 → Base64
        # vector = np.random.rand(1024).astype(np.float32)  # 1024维向量
        # base64_str = base64.b64encode(vector.tobytes()).decode()

        # Base64 → 向量
        # decoded_bytes = base64.b64decode(base64_str)
        # restored_vector = np.frombuffer(decoded_bytes, dtype=np.float32)

        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="jina_04afb1e9a75e41d9acde9357d01f2c6fTAk8UbRt3usi9AA8y0NCLq1-Kr9l"
        )
        response = client.embeddings.create(
            input=["Your text string goes here"],
            model="bge-code-v1",
            dimensions=1024,
            encoding_format="base64"
        )
        print(len(response.data[0].embedding))
        print(response.data[0].embedding)

if __name__ == '__main__':
    unittest.main()
