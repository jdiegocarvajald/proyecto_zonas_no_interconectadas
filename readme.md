# PASOS PARA CORRER EL PROYECTO

## Python

1. Primero debes verificar que tienes instalada la última versión de pyhton en tu entorno local.
2. Luego, debes instalar todas las librerías usadas en el proyecto con el siguiente comando en la terminal de tu IDE:
        pip install -r requirements.txt
3. Para la conexión a la base de datos te recomendamos usar PGAdmin, el gestor de base de datos de PostgreSQL.
   En la raíz del proyecto podrás encontrar el archivo "Backup_Proyecto_Final.sql" con el backup de la base de datos original; 
   en PGAdmin deberás crear una base de datos y luego usar la opción "Restore" para importar el archivo "Backup_Proyecto_Final.sql" 
   y así importar la base de datos con sus esquemas y la vista generada para el análisis de datos posterior.
4. Al tener las librerías del archivo "requirements.txt" instaladas y la base de datos importada corriendo en local necesitarás 
   conectar el proyecto a la base de datos, para esto en la variable "engine" deberás cambiar el nombre de la base de datos y la
   contraseña que hay por defecto y reemplazarlas por tus propias credenciales.
5. Ahora, para levantar el servicio deberás abrir una terminal en tu IDE y ejecutar el siguiente comando:
        streamlit run 3_Streamlit.py
   este comando abrirá una ventana nueva en tu navegador predeterminado en tu entorno y podrás ver el proyecto funcional.
6. En caso de que te aparezcan errore en el navegador, con una simple búsqueda de ayuda en Chatgpt puedes resolver 
   el problema.

## Notas:

1. en la raíz del proyecto podrás encontrar todos los arivos necesarios para que puedas correr el proyecto, no necesitas
   ningún otro archivo externo.
2. También podrás encontrar los archivos de referencia ".csv" y ".xlsx".

