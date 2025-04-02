FROM python:3.12.5-bookworm

WORKDIR /root

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install mlflow
RUN pip3 install scikit-learn

COPY . /root/

EXPOSE 8050

CMD ["python", "app/app.py"]
