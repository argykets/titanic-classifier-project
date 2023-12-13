FROM python:3.9.9

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' argykets

WORKDIR /opt/titanic-classifier-api

ARG PIP_EXTRA_INDEX_URL

# Install requirements, including from Gemfury
ADD ./titanic-classifier-api /opt/titanic-classifier-api/
ADD ./titanic-classifier-package /opt/titanic-classifier-package/
RUN pip install --upgrade pip
RUN pip install -r /opt/titanic-classifier-api/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}/opt/titanic-classifier-package/classification_model"
RUN chmod +x /opt/titanic-classifier-api/run.sh
RUN chown -R argykets:argykets ./

USER argykets

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
