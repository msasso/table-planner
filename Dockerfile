FROM python:3.8

WORKDIR /code

ENV \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

RUN pip install "poetry==1.1"
#RUN poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml /code/
RUN mkdir -p /data && poetry config virtualenvs.create false && poetry install --no-interaction 

COPY src/ /code/

CMD [ "python", "/code/planner.py", "/data/input.csv" ]
