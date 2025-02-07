FROM python:3.11-slim
#set workdir in docker
WORKDIR /code 
#copy requirments.txt file
COPY ./requirements.txt ./

RUN apt-get update && apt-get install git -y && apt-get install curl -y

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/code/src
EXPOSE 8001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]