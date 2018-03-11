#ifndef PGM_IO_H
#define PGM_IO_H
int read_pgm_image(char *infilename, unsigned char **image, int *rows,int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows, int cols, char *comment, int maxval);
int read_ppm_image(char *infilename, unsigned char **image_red, unsigned char **image_grn, unsigned char **image_blu, int *rows, int *cols);
int write_ppm_image(char *outfilename, unsigned char *image_red, unsigned char *image_grn, unsigned char *image_blu, int rows, int cols, char *comment, int maxval);
#endif