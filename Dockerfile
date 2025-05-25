# Usa una imagen base de Python. Python 3.9 o 3.10 son buenas opciones estables.
FROM python:3.9-slim-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todo el contenido de tu repositorio al directorio de trabajo en el contenedor
# Esto copiará tanto 'backend/' como 'frontend/' si están en la raíz de tu repo
COPY . /app

# Establece el directorio de trabajo específico para tu backend
# Aquí es donde 'app.py' y 'requirements.txt' se encuentran
WORKDIR /app/backend

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Comando para iniciar tu aplicación Flask con Gunicorn
# --bind 0.0.0.0:7860 es CRUCIAL para Hugging Face Spaces
# 'app:app' significa que tu archivo es 'app.py' y la instancia de Flask es 'app'
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
