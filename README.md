# iama

.add_and_commit
Es necesario para .push_and_dockerbuild

.docker_run [command]
ese bash facilita la ejecucion del docker. Se puede hacer:
    .docker_run -> para ejecutar main.py
    .docker_run m -> para ejecutar main.py
    .docker_run main -> para ejecutar main.py

o para training.
    .docker_run t -> para ejecutar training.py
    .docker_run train -> para ejecutar training.py

.push_and_dockerbuild
Construye la imagen Docker con BuildKit.
Hace commit de los cambios en Git (si existen) con el mensaje proporcionado.
Envia la rama main al repositorio remoto.

Asegurense de que ambos archivos tengan permiso de ejecucion si usan linux:
    sudo chmod +x .add_and_commit .docker_run .push_and_dockerbuild

o usen alguna herramienta como git bash
    1) Abren una terminal de GitBash en la carpeta donde lo clonaron.
    2) Dan permiso de ejecucion a ambos archivos
        (chmod +x .add_and_commit .docker_run .push_and_dockerbuild)
    3) Enjoy!
