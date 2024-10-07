FROM python:3.11.4-bookworm

WORKDIR /base

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install numpy
RUN pip3 install scikit-learn

# Testing module
RUN pip3 install dash[testing]
RUN pip3 install pytest
RUN pip3 install pytest-depends

# add mlflow
RUN pip3 install mlflow

COPY . /base

CMD tail -f /dev/null
