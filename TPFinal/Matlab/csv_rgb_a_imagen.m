%Autor: Federico Perez
%Funcion que toma los archivos .csv que corresponden
%a los 3 colores rgb de una imagen, los vuelve a componer en
%1 sola y muestra la imagen. Los archivos deben tener el mismos
%rango NxM. Los parametros son los nombres de los archivos.
function csv_rgb_a_imagen(rojo, verde, azul)
%leo los archivos
disp('Leyendo archivos...');
r = csvread(rojo);
v = csvread(verde);
a = csvread(azul);
%compongo la imagen
disp('Componiendo imagen.');
foto_original = cat(3,r,v,a);
foto_original = uint8(foto_original);
image(foto_original);
disp('Terminado.');
end