#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Kernel Jacobi (mise à jour des points intérieurs uniquement)
__global__ void jacobi_step(double* a_new, const double* a, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix > 0 && ix < nx-1 && iy > 0 && iy < ny-1) {
        int idx = iy * nx + ix;
        a_new[idx] = 0.25 * (
            a[idx - nx] +   // voisin du haut
            a[idx + nx] +   // voisin du bas
            a[idx - 1 ] +   // voisin de gauche
            a[idx + 1 ]     // voisin de droite
        );
    }
}

// Kernel pour appliquer les conditions périodiques en Y
__global__ void apply_periodic_bc(double* a, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < nx) {
        int top    = 0 * nx + ix;
        int bottom = (ny-1) * nx + ix;
        int first  = 1 * nx + ix;
        int last   = (ny-2) * nx + ix;
        a[top]    = a[last];
        a[bottom] = a[first];
    }
}

// Fonction pour afficher l'état de la grille côté host
void printState(const double* a, int nx, int ny) {
    for (int iy = 0; iy < ny; iy++) {
        int base = iy * nx;
        for (int ix = 0; ix < nx; ix++) {
            printf("%1f ", a[base + ix]);
        }
        printf("\n");
    }
    printf("\n");
}

// Initialisation sur l'host : bords à left/right, reste à zéro
void initialize_host(double* a, int nx, int ny, double left, double right) {
    for (int iy = 0; iy < ny; iy++) {
        int base = iy * nx;
        a[base] = left;
        a[base + nx - 1] = right;
        for (int ix = 1; ix < nx-1; ix++) {
            a[base + ix] = 0.0;
        }
    }
}

// Foncteur Thrust pour calculer |a_new - a_old| (convergence)
struct max_abs_diff {
    __host__ __device__
    double operator()(const thrust::tuple<double,double>& t) const {
        return fabs(thrust::get<0>(t) - thrust::get<1>(t));
    }
};

int main(int argc, char* argv[]) {
    // On vérifie les arguments de la ligne de commande
    if (argc != 6) {
        fprintf(stderr, "Usage: %s nx ny max_iter tol print_flag\n", argv[0]);
        fprintf(stderr, "  tol: tolérance de convergence (ex : 1e-6)\n");
        fprintf(stderr, "  print_flag: 1 pour afficher l'état final, 0 sinon\n");
        return EXIT_FAILURE;
    }
    int nx         = atoi(argv[1]);
    int ny         = atoi(argv[2]);
    int max_iter   = atoi(argv[3]);
    double tol     = atof(argv[4]);
    int print_flag = atoi(argv[5]);

    const int N = nx * ny;
    size_t bytes = N * sizeof(double);

    // Mesure du temps global (tout le code)
    clock_t global_start = clock();

    // Allocation des tableaux côté host
    double* h_a     = (double*)malloc(bytes);
    double* h_a_new = (double*)malloc(bytes);
    initialize_host(h_a,     nx, ny, 1.0, 1.0);
    initialize_host(h_a_new, nx, ny, 1.0, 1.0);

    // Allocation côté device
    double *d_a, *d_a_new;
    cudaMalloc(&d_a,     bytes);
    cudaMalloc(&d_a_new, bytes);

    // On copie les données initiales sur le GPU
    cudaMemcpy(d_a,     h_a,     bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_new, h_a_new, bytes, cudaMemcpyHostToDevice);

    // Paramètres pour le lancement des kernels
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    // Chronométrage CUDA (calcul pur)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Boucle de Jacobi jusqu’à la convergence ou le max d’itérations
    double error = tol + 1.0;
    int iter = 0;
    while (error > tol && iter < max_iter) {
        // 1) Une étape Jacobi
        jacobi_step<<<grid, block>>>(d_a_new, d_a, nx, ny);
        // 2) Application des conditions périodiques en Y
        apply_periodic_bc<<<(nx + 255) / 256, 256>>>(d_a_new, nx, ny);

        // 3) Calcul de l’erreur max via Thrust
        thrust::device_ptr<double> ptr_new(d_a_new);
        thrust::device_ptr<double> ptr_old(d_a);
        error = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new, ptr_old)),
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new + N, ptr_old + N)),
            max_abs_diff(),
            0.0,
            thrust::maximum<double>());

        // 4) On échange les pointeurs pour la prochaine itération
        std::swap(d_a, d_a_new);

        iter++;
    }

    // Stop chronométrage CUDA
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // On recopie le résultat final sur l'host
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    // Affichage du nombre d’itérations, erreur finale, temps CUDA et temps total
    printf("Solveur Jacobi CUDA convergé en %d itérations (erreur = %.3e)\n", iter, error);
    printf("Temps passé (calcul CUDA) : %.6f s\n", milliseconds / 1000.0);

    // Mesure du temps global
    double global_time = (double)(clock() - global_start) / CLOCKS_PER_SEC;
    printf("Temps total du programme (tout inclus) : %.6f s\n", global_time);

    if (print_flag == 1) {
        printf("État final :\n");
        printState(h_a, nx, ny);
    }

    // Libération des ressources
    free(h_a);
    free(h_a_new);
    cudaFree(d_a);
    cudaFree(d_a_new);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
