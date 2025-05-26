#include "laplace2d.h"
#include <stdlib.h>
#include <stdio.h>

void initialize(double *a, int nx, int ny, double left, double right) {
    for (int iy = 0; iy < ny; iy++) {
        int base = iy * nx;
        a[base] = left; // bord gauche
        a[base + nx - 1] = right; // bord droit
        for (int ix = 1; ix < nx - 1; ix++) {
            a[base + ix] = 0.0;
        }
    }
}

void jacobi_step(double *a_new, double *a, int nx, int ny) {
    for (int iy = 1; iy < ny - 1; iy++) {
        int row = iy * nx;
        int row_up = (iy - 1) * nx;
        int row_down = (iy + 1) * nx;
        for (int ix = 1; ix < nx - 1; ix++) {
            a_new[row + ix] = 0.25 * (
                a[row_up + ix] +     // haut
                a[row_down + ix] +   // bas
                a[row + (ix - 1)] +  // gauche
                a[row + (ix + 1)]    // droite
            );
        }
    }
}

void apply_periodic_bc(double *a, int nx, int ny) {
    int base_last = (ny - 2) * nx;
    int base_first = 1 * nx;
    // Ligne du haut
    for (int ix = 0; ix < nx; ix++) {
        a[ix] = a[base_last + ix];
    }
    // Ligne du bas
    int base_bottom = (ny - 1) * nx;
    for (int ix = 0; ix < nx; ix++) {
        a[base_bottom + ix] = a[base_first + ix];
    }
}

void swap(double **a, double **b) {
    double *temp = *a;
    *a = *b;
    *b = temp;
}

void printState(double *a, int nx, int ny) {
    for (int iy = 0; iy < ny; iy++) {
        int base = iy * nx;
        for (int ix = 0; ix < nx; ix++) {
            printf("%lf ", a[base + ix]);
        }
        putchar('\n');
    }
    putchar('\n');
}
