#ifndef HYSTERESIS_H
#define HYSTERESIS_H

void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval,int cols);

void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols,	float tlow, float thigh, unsigned char *edge, int rank, int num_processes);

void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char *result);

#endif
