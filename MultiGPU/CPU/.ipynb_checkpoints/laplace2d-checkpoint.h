#ifndef LAPLACE2D_H
#define LAPLACE2D_H

void initialize(double *a, int nx, int ny, double left, double right);
void jacobi_step(double *a_new, double *a, int nx, int ny);
void apply_periodic_bc(double *a, int nx, int ny);
void swap(double **a, double **b);
void printState(double *a, int nx, int ny);

#endif