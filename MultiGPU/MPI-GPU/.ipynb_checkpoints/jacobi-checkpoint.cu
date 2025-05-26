// jacobi_mpi_cuda.cu
#include <mpi.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define IDX(iy, ix, nx) ((iy)*(nx)+(ix))
#define BLOCK 16

// Affichage final de la grille (côté host)
void printState(const double* a, int nx, int ny) {
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++)
            printf("%.6f ", a[iy * nx + ix]);
        printf("\n");
    }
    printf("\n");
}

// Initialisation des bords (Dirichlet à 1), intérieur à 0
__global__
void init_boundaries(double *a, double *a_new, int nx, int ghost_ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ghost_ny) return;
    if (ix == 0 || ix == nx-1) {
        a[IDX(iy,ix,nx)]     = 1.0;
        a_new[IDX(iy,ix,nx)] = 1.0;
    } else {
        a[IDX(iy,ix,nx)]     = 0.0;
        a_new[IDX(iy,ix,nx)] = 0.0;
    }
}

// Calcul d'une itération Jacobi sur la zone intérieure (hors bords/halos)
__global__
void jacobi_kernel(const double *a, double *a_new, int nx, int ghost_ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < 1 || ix >= nx-1 || iy < 1 || iy >= ghost_ny-1) return;
    int idx = IDX(iy,ix,nx);
    a_new[idx] = 0.25 * (
        a[idx - nx] +   // voisin du haut
        a[idx + nx] +   // voisin du bas
        a[idx - 1 ] +   // voisin de gauche
        a[idx + 1 ]     // voisin de droite
    );
}

// Foncteur Thrust pour obtenir le max des différences absolues
struct max_abs_diff {
    __host__ __device__
    double operator()(const thrust::tuple<double,double>& t) const {
        return fabs(thrust::get<0>(t) - thrust::get<1>(t));
    }
};

// Fonction pour échanger deux pointeurs
static inline void swap_ptrs(double **p, double **q) {
    double *t=*p; *p=*q; *q=t;
}

int main(int argc, char **argv) {
    // Mesure du temps global (tout inclus, allocations, MPI etc.)
    clock_t global_start = clock();

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    cudaSetDevice(rank % num_gpus);

    if (argc != 6) {
        if (!rank)
            fprintf(stderr,"Usage: %s nx ny max_iter tol print_flag\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int nx         = atoi(argv[1]);
    int ny         = atoi(argv[2]);
    int max_iter   = atoi(argv[3]);
    double tol     = atof(argv[4]);
    int print_flag = atoi(argv[5]);

    // On répartit le boulot en bandes horizontales, même si ny-2 n'est pas multiple de size
    int work_ny = ny - 2; // lignes internes (on ne compte pas les bords haut/bas)
    int chunk_size_low  = work_ny / size;
    int chunk_size_high = chunk_size_low + 1;
    int num_ranks_low   = size * chunk_size_low + size - work_ny;
    int local_ny, iy_start_global;
    if (rank < num_ranks_low) {
        local_ny = chunk_size_low;
        iy_start_global = rank * chunk_size_low + 1;
    } else {
        local_ny = chunk_size_high;
        iy_start_global = num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
    }
    // +2 pour les halos haut et bas
    int ghost_ny = local_ny + 2;

    // Allocation mémoire sur le GPU
    size_t bytes = ghost_ny * nx * sizeof(double);
    double *d_a, *d_a_new;
    cudaMalloc(&d_a,     bytes);
    cudaMalloc(&d_a_new, bytes);

    // Initialisation des bords sur le GPU
    dim3 block(BLOCK, BLOCK), grid((nx+BLOCK-1)/BLOCK, (ghost_ny+BLOCK-1)/BLOCK);
    init_boundaries<<<grid, block>>>(d_a, d_a_new, nx, ghost_ny);
    cudaDeviceSynchronize();

    // Chronométrage MPI+GPU pur (hors allocations et init)
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int prev = (rank-1+size)%size, next = (rank+1)%size;
    double error = tol + 1.0;
    int iter = 0;

    thrust::device_ptr<double> ptr_new(d_a);
    thrust::device_ptr<double> ptr_old(d_a_new);
    size_t local_N = ghost_ny * nx;

    while (error > tol && iter < max_iter) {
        // Échanges des halos haut/bas (MPI_Sendrecv sur la device ptr si CUDA-aware)
        MPI_Sendrecv(d_a + IDX(1,0,nx), nx, MPI_DOUBLE, prev, 0,
                     d_a + IDX(ghost_ny-1,0,nx), nx, MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(d_a + IDX(local_ny,0,nx), nx, MPI_DOUBLE, next, 1,
                     d_a + IDX(0,0,nx), nx, MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        jacobi_kernel<<<grid,block>>>(d_a, d_a_new, nx, ghost_ny);
        cudaDeviceSynchronize();
        swap_ptrs(&d_a,&d_a_new);

        // Calcul de l’erreur max locale puis reduction MPI
        error = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new, ptr_old)),
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new + local_N, ptr_old + local_N)),
            max_abs_diff(),
            0.0,
            thrust::maximum<double>());
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        iter++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Affichage du temps scientifique (MPI+multiGPU) et du temps total réel
    if (!rank) {
        printf("Solveur Jacobi MPI+multiGPU convergé en %d itérations (erreur = %.3e)\n", iter, error);
        printf("Temps calcul MPI+multiGPU (hors init): %.6f s\n", t1-t0);

        // Mesure du temps global (vraiment tout inclus)
        double global_time = (double)(clock() - global_start) / CLOCKS_PER_SEC;
        printf("Temps total du programme (allocs, MPI, init, calcul, etc.): %.6f s\n", global_time);
    }

    // Option pour afficher l’état final de la grille (rank 0 seulement)
    if (print_flag) {
        double *local_result = (double*)malloc(local_ny*nx*sizeof(double));
        cudaMemcpy(local_result, d_a+IDX(1,0,nx), local_ny*nx*sizeof(double), cudaMemcpyDeviceToHost);
        if (rank == 0) {
            double *full = (double*)malloc(nx*ny*sizeof(double));
            memcpy(full + iy_start_global*nx, local_result, local_ny*nx*sizeof(double));
            for (int r=1; r<size; r++){
                int r_ny, r_iy_start;
                if (r < num_ranks_low) {
                    r_ny = chunk_size_low;
                    r_iy_start = r * chunk_size_low + 1;
                } else {
                    r_ny = chunk_size_high;
                    r_iy_start = num_ranks_low * chunk_size_low + (r - num_ranks_low) * chunk_size_high + 1;
                }
                MPI_Recv(full + r_iy_start*nx, r_ny*nx, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // Recopie les lignes 1 et ny-2 sur les bords (périodique)
            memcpy(full,               full + nx,         nx * sizeof(double));            
            memcpy(full + (ny-1)*nx,   full + (ny-2)*nx,  nx * sizeof(double));            

            printf("État final:\n");
            printState(full, nx, ny);
            free(full);
        } else {
            MPI_Send(local_result, local_ny*nx, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        free(local_result);
    }

    cudaFree(d_a);
    cudaFree(d_a_new);
    MPI_Finalize();
    return 0;
}
