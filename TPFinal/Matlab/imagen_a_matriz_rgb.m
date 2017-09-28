%Autor: Federico Perez
%Funcion que descompone una imagen RGB en tres matrices con los
%colores separados, y luego escribe dichas matrices en archivos
%con formato .csv.
%el parametro que recibe es el nombre de la imagen a procesar
function imagen_a_matriz_rgb(nombre)

disp('Cargando imagen...')

imagen = imread(nombre);
disp('Separando colores...')
%separo la matriz tridimensional en bidimensional
rojo = imagen(:,:,1); 
verde = imagen(:,:,2); 
azul = imagen(:,:,3);

%escribo los archivos en formato csv
csvwrite('rojo.csv',rojo);
disp('Creado rojo.csv')
csvwrite('verde.csv',verde);
disp('Creado verde.csv')
csvwrite('azul.csv',azul);
disp('Creado azul.csv')

clearvars imagen;
disp('Terminado.')
end