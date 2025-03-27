#!/bin/bash

# 检查服务器健康状态
echo "检查服务器健康状态..."
curl -X GET http://localhost:8000/health

echo ""
echo "获取模型信息..."
curl -X GET http://localhost:8000/model_info

echo ""
echo "发送预测请求..."
curl -X POST \
  http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": [
      [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
      [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
      [[1.3, 1.4, 1.5], [1.6, 1.7, 1.8]]
    ],
    "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  }'

# 使用更大的图像发送请求（更接近实际用例）
echo ""
echo "使用较大图像发送请求..."
python -c '
import json
import numpy as np
import requests

# 创建一个3x224x224的随机图像
image = np.random.rand(3, 64, 64).tolist()
state = np.random.rand(7).tolist()

payload = {
    "image": image,
    "state": state
}

# 发送请求
response = requests.post("http://localhost:8000/predict", json=payload)
print(f"状态码: {response.status_code}")
print(f"响应: {json.dumps(response.json(), indent=2)}")
' 