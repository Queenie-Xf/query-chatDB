FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# 确保安装pymongo
RUN pip install --no-cache-dir pymongo

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/
COPY data/ ./data/


# 确保目录结构存在
RUN mkdir -p /app/data

# 设置环境变量
ENV PYTHONPATH=/app

# 创建启动脚本
RUN echo '#!/bin/bash\n\
echo "Initializing MongoDB..."\n\
python /app/init_mongo.py\n\
echo "Starting API server..."\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000\n' > /app/start.sh

RUN chmod +x /app/start.sh

# 运行启动脚本
CMD ["/app/start.sh"]