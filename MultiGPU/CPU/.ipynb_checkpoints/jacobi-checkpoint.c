#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

void initialize(double *a, int nx, int ny, double left, double right);
void jacobi_step(double *a_new, double *a, int nx, int ny);
void apply_periodic_bc(double *a, int nx, int ny);
void printState(double *a, int nx, int ny);

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s nx ny max_iter tol affichage\n", argv[0]);
        return 1;
    }

    // Parsing arguments (une seule fois)
    int    nx         = atoi(argv[1]);
    int    ny         = atoi(argv[2]);
    int    max_iter   = atoi(argv[3]);
    double tol        = atof(argv[4]);
    bool   affichage  = atoi(argv[5]);
    int    N          = nx * ny;  // Calculé une fois

    // Allocation (une seule fois)
    double *a     = malloc(N * sizeof(double));
    double *a_new = malloc(N * sizeof(double));
    if (!a || !a_new) {
        perror("malloc");
        return 1;
    }

    // Initialisation
    initialize(a, nx, ny, 1.0, 1.0);
    initialize(a_new, nx, ny, 1.0, 1.0);
    if (affichage) {
        printState(a, nx, ny);
    }

    double t_start = omp_get_wtime();

    double error = tol + 1.0;
    int iter = 0;

    // Déclarations en dehors de la boucle
    int iy, ix;

    while (error > tol && iter < max_iter) {
        error = 0.0;

        jacobi_step(a_new, a, nx, ny);
        apply_periodic_bc(a_new, nx, ny);

        for (iy = 1; iy < ny - 1; iy++) {
            for (ix = 1; ix < nx - 1; ix++) {
                int tot = iy*nx;
                double diff = fabs(a_new[tot + ix] - a[tot + ix]);
                if (diff > error) error = diff;
            }
        }

        // échange direct des pointeurs a et a_new
        double *tmp = a;
        a = a_new;
        a_new = tmp;

        iter++;
    }

    double t_end   = omp_get_wtime();
    double elapsed = t_end - t_start;

    if (affichage) {
        printState(a, nx, ny);
    }

    // Résultats
    printf("Jacobi solver stopped after %d iterations\n", iter);
    if (error <= tol)
        printf("Converged (error = %.6e ≤ tol = %.6e)\n", error, tol);
    else
        printf("Reached max_iter = %d (error = %.6e > tol = %.6e)\n",
               max_iter, error, tol);

    printf("Elapsed time   : %.6f seconds\n", elapsed);
    printf("Values at corners: [0,0]=%.2f  [0,%d]=%.2f\n",
           a[0], nx-1, a[nx-1]);

    int ic = ny/2, jc = nx/2;
    printf("Value at center [%d,%d] = %.6f\n",
           ic, jc, a[ic*nx + jc]);

    free(a);
    free(a_new);
    return 0;
}
