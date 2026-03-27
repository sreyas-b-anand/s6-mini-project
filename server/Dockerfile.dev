FROM python:3.11

WORKDIR /server

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY .env .

RUN python scripts/download_model.py

EXPOSE 8000

CMD ["uvicorn", "main:app" , "--reload"]