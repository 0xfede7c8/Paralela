#ifndef HYSTERESIS_H
#define HYSTERESIS_H

#define VERBOSE 1
#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval,int cols);

void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,	float tlow, float thigh, unsigned char *edge);

void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char **result);

__global__
void cuda_non_max_supp(  short *mag,
                    short *gradx,
                    short *grady,
                    int nrows,
                    int ncols,
                    unsigned char *result);

#endif
