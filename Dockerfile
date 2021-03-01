FROM python:3.8.5-buster

WORKDIR /workdir
EXPOSE 8501
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get -y install python3-pip
RUN apt-get install git -y
RUN pip3 install streamlit
RUN pip3 install torch
RUN pip3 install pdfminer
RUN pip3 install tensorflow-hub
RUN pip3 install tensorflow
RUN pip3 install preshed
RUN pip3 install bert-extractive-summarizer
# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

CMD ["streamlit", "run", "app.py"]