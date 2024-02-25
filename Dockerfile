FROM python:3.11-slim

WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/RajatJacob/lindt-home-of-chocolate-bot .
ADD . .

RUN pip3 install streamlit
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader punkt

ADD . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["python3", "-m", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
