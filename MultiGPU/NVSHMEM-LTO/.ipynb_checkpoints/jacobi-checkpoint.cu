#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>  // Ajouté pour la mesure du temps total global
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>

#define IDX(iy, ix, nx) ((iy)*(nx)+(ix))
#define BLOCK 16

// Petite fonction pour afficher la grille à la fin (côté host)
void printState(const double* a, int nx, int ny) {
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++)
            printf("%.6f ", a[iy * nx + ix]);
        printf("\n");
    }
    printf("\n");
}

// J'initialise les bords à 1 et tout le reste à 0
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

// Le kernel Jacobi sur mon sous-domaine local
__global__
void jacobi_kernel(const double *a, double *a_new, int nx, int ghost_ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < 1 || ix >= nx-1 || iy < 1 || iy >= ghost_ny-1) return;
    int idx = IDX(iy,ix,nx);
    a_new[idx] = 0.25 * (
        a[idx - nx] +   // le point au-dessus
        a[idx + nx] +   // le point en dessous
        a[idx - 1 ] +   // à gauche
        a[idx + 1 ]     // à droite
    );
}

// Pour choper le max des différences entre deux vecteurs (pour tester la convergence)
struct max_abs_diff {
    __host__ __device__
    double operator()(const thrust::tuple<double,double>& t) const {
        return fabs(thrust::get<0>(t) - thrust::get<1>(t));
    }
};

// Permet d'échanger les pointeurs simplement (pratique à la fin d'une itération)
void swap_ptrs(double **p, double **q) {
    double *t = *p; *p = *q; *q = t;
}

int main(int argc, char **argv) {
    // 1) Chronomètre global pour mesurer tout le code (allocations, init, calcul, affichage, etc)
    
    clock_t global_start = clock();
    double t_nvshmem_prep_start = MPI_Wtime();

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Attribution du GPU local pour chaque processus MPI sur le nœud
    int local_rank;
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    cudaSetDevice(local_rank);
    cudaFree(0);
    MPI_Comm_free(&local_comm);

    if (argc != 6) {
        if (!rank) fprintf(stderr, "Usage: %s nx ny max_iter tol print_flag\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    double tol = atof(argv[4]);
    int print_flag = atoi(argv[5]);

    // Je découpe la grille pour répartir le boulot de façon équilibrée 
    int work_ny = ny - 2;
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
    int ghost_ny = local_ny + 2;

    // J'ai besoin de connaître la taille de ghost_ny de chaque PE (pour bien placer les halos)
    int* ghost_nys = (int*)malloc(size * sizeof(int));
    int* local_nys = (int*)malloc(size * sizeof(int));
    MPI_Allgather(&local_ny, 1, MPI_INT, local_nys, 1, MPI_INT, MPI_COMM_WORLD);
    for(int i=0; i<size; i++) ghost_nys[i] = local_nys[i] + 2;

    // Je prépare le heap NVSHMEM à la taille maximale requise par tous les PE (nécessaire pour la symétrie mémoire)
    int max_ghost_ny;
    MPI_Allreduce(&ghost_ny, &max_ghost_ny, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    size_t bytes = max_ghost_ny * nx * sizeof(double);
    size_t required_heap = 2 * bytes * 1.1;
    if (!getenv("NVSHMEM_SYMMETRIC_SIZE")) {
        char buf[64];
        sprintf(buf, "%zu", required_heap);
        setenv("NVSHMEM_SYMMETRIC_SIZE", buf, 1);
    }

    // Initialisation de NVSHMEM (avec le communicateur MPI)
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    double t_nvshmem_prep_end = MPI_Wtime();

    double t_nvshmem_init_start = MPI_Wtime();
    if (rank == 0)
    printf("[NVSHMEM] Temps d'init avant NVSHMEM : %.6fs\n", t_nvshmem_prep_end - t_nvshmem_prep_start);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    double t_nvshmem_init_end = MPI_Wtime();
    if (rank == 0)
    printf("[NVSHMEM] Temps d'init NVSHMEM : %.6fs\n", t_nvshmem_init_end - t_nvshmem_init_start);

    nvshmem_barrier_all(); 

    // 2) Chrono "setup+calcul+affichage" : juste après init NVSHMEM/barrier
    double t_after_nvshmem_init = MPI_Wtime();

    // J'alloue la mémoire sur le heap symétrique NVSHMEM (taille identique partout)
    double *d_a     = (double*)nvshmem_malloc(bytes);
    double *d_a_new = (double*)nvshmem_malloc(bytes);

    // Je lance l'init des bords sur le device
    dim3 block(BLOCK, BLOCK);
    dim3 grid((nx + BLOCK-1)/BLOCK, (ghost_ny + BLOCK-1)/BLOCK);
    init_boundaries<<<grid, block>>>(d_a, d_a_new, nx, ghost_ny);
    cudaDeviceSynchronize();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 3) Chrono calcul pur (boucle Jacobi uniquement)
    nvshmem_barrier_all();
    double t_calc0 = MPI_Wtime();

    int prev = (rank - 1 + size) % size;
    int next = (rank + 1) % size;
    double error = tol + 1.0;
    int iter = 0;

    while (error > tol && iter < max_iter) {
        // Ici, j'envoie mes halos aux bons offsets dans le buffer du voisin,
        // car chaque PE peut avoir un ghost_ny différent (découpage non uniforme).
        // - J'envoie ma DERNIÈRE ligne réelle dans le halo BAS du voisin du BAS.
        nvshmem_double_put(
            d_a + IDX(ghost_nys[next]-1, 0, nx),     // Destination: halo bas du voisin next
            d_a + IDX(local_ny, 0, nx),              // Source: ma dernière ligne réelle
            nx, next);

        // - J'envoie ma PREMIÈRE ligne réelle dans le halo HAUT du voisin du HAUT.
        nvshmem_double_put(
            d_a + IDX(0, 0, nx),                     // Destination: halo haut du voisin prev
            d_a + IDX(1, 0, nx),                     // Source: ma première ligne réelle
            nx, prev);

        // Un fence pour garantir la visibilité mémoire côté remote avant le calcul
        nvshmem_fence();
        // Barrière pour s'assurer que tout le monde a bien reçu les halos avant de calculer
        nvshmem_barrier_all();

        // Je lance l'itération Jacobi sur mon sous-domaine local
        jacobi_kernel<<<grid, block, 0, stream>>>(d_a, d_a_new, nx, ghost_ny);
        cudaStreamSynchronize(stream);

        // Calcul de l'erreur max locale (avec thrust) puis on fait le allreduce pour la convergence globale
        thrust::device_ptr<double> ptr_new(d_a_new);
        thrust::device_ptr<double> ptr_old(d_a);
        error = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new, ptr_old)),
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new + ghost_ny*nx, ptr_old + ghost_ny*nx)),
            max_abs_diff(),
            0.0,
            thrust::maximum<double>());
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Permet d'échanger les pointeurs simplement (pratique à la fin d'une itération)
        swap_ptrs(&d_a, &d_a_new);
        iter++;
    }

    nvshmem_barrier_all();
    double t_calc1 = MPI_Wtime(); // Fin du calcul pur

    // Fin du 2e chrono (setup+calcul+affichage)
    nvshmem_barrier_all();
    double t_after_all = MPI_Wtime();

    // Affichage des 3 temps sur le rang 0
    if (rank == 0) {
        printf("[NVSHMEM] Converged in %d iterations | Error: %.2e\n", iter, error);
        printf("Temps de calcul Jacobi (seulement boucle)         : %.6fs\n", t_calc1 - t_calc0);
        printf("Temps setup+calcul+affichage (après init SHMEM)   : %.6fs\n", t_after_all - t_after_nvshmem_init);
        double global_time = (double)(clock() - global_start) / CLOCKS_PER_SEC;
        printf("Temps total du programme (tout compris)           : %.6fs\n", global_time);
    }

    // Je rassemble tout et j'affiche le résultat final (comme avant)
    if (print_flag) {
        double* local_result = (double*)malloc(local_ny * nx * sizeof(double));
        cudaMemcpy(local_result, d_a + IDX(1,0,nx), local_ny * nx * sizeof(double), cudaMemcpyDeviceToHost);

        if (rank == 0) {
            double* full = (double*)malloc(ny * nx * sizeof(double));
            memcpy(full + iy_start_global * nx, local_result, local_ny * nx * sizeof(double));
            for (int r = 1; r < size; r++) {
                int r_ny, r_offset;
                if (r < num_ranks_low) {
                    r_ny = chunk_size_low;
                    r_offset = r * chunk_size_low + 1;
                } else {
                    r_ny = chunk_size_high;
                    r_offset = num_ranks_low * chunk_size_low + (r - num_ranks_low) * chunk_size_high + 1;
                }
                MPI_Recv(full + r_offset * nx, r_ny * nx, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // Recopie les bords (y=0 et y=ny-1)
            memcpy(full, full + nx, nx * sizeof(double));
            memcpy(full + (ny-1)*nx, full + (ny-2)*nx, nx * sizeof(double));
            printf("État final :\n");
            printState(full, nx, ny);
            free(full);
        } else {
            MPI_Send(local_result, local_ny * nx, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        free(local_result);
    }

    if(d_a) nvshmem_free(d_a);
    if(d_a_new) nvshmem_free(d_a_new);
    cudaStreamDestroy(stream);
    nvshmem_finalize();
    MPI_Finalize();
    free(ghost_nys);
    free(local_nys);
    return 0;
}
