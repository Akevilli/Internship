FROM apache/airflow:3.1.5

USER root
RUN mkdir -p /opt/airflow/dvc_repo && chown -R airflow: /opt/airflow/dvc_repo

USER airflow
WORKDIR /opt/airflow/dvc_repo

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

COPY --chown=airflow:airflow .dvc /opt/airflow/dvc_repo/.dvc
COPY --chown=airflow:airflow data.dvc /opt/airflow/dvc_repo/