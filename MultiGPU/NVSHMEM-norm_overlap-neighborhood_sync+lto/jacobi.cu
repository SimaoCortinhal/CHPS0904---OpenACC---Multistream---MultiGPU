#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>

#define IDX(iy, ix, nx) ((iy)*(nx)+(ix))
#define BLOCK 16

void printState(const double* a, int nx, int ny) {
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++)
            printf("%.6f ", a[iy * nx + ix]);
        printf("\n");
    }
    printf("\n");
}

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

// *** NOUVEAU ***
// Jacobi sur la "zone intérieure" (aucune dépendance au halo)
__global__
void jacobi_inner_kernel(const double *a, double *a_new, int nx, int ghost_ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    // On évite la première/dernière colonne, et les deux premières et deux dernières lignes
    if (ix < 1 || ix >= nx-1 || iy < 2 || iy >= ghost_ny-2) return;
    int idx = IDX(iy,ix,nx);
    a_new[idx] = 0.25 * (
        a[idx - nx] +   // au-dessus
        a[idx + nx] +   // en dessous
        a[idx - 1 ] +   // à gauche
        a[idx + 1 ]     // à droite
    );
}

// Jacobi sur la "zone bord haut et bas" (dépend du halo, à exécuter après synchro)
__global__
void jacobi_border_kernel(const double *a, double *a_new, int nx, int ghost_ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    // Bord haut = ligne 1, bord bas = ligne ghost_ny-2
    // On ne traite pas la toute première ni la toute dernière ligne (fantôme)
    if (ix < 1 || ix >= nx-1) return;
    // Lignes frontalières
    for (int iy = 1; iy < ghost_ny-1; iy += ghost_ny-3) { // iy = 1 et iy = ghost_ny-2
        int idx = IDX(iy,ix,nx);
        a_new[idx] = 0.25 * (
            a[idx - nx] +
            a[idx + nx] +
            a[idx - 1 ] +
            a[idx + 1 ]
        );
    }
}

struct max_abs_diff {
    __host__ __device__
    double operator()(const thrust::tuple<double,double>& t) const {
        return fabs(thrust::get<0>(t) - thrust::get<1>(t));
    }
};

void swap_ptrs(double **p, double **q) {
    double *t = *p; *p = *q; *q = t;
}

__global__ void syncneighborhood_kernel(int my_pe, int num_pes, uint64_t* sync_arr, uint64_t counter) {
    int next_rank = (my_pe + 1) % num_pes;
    int prev_rank = (my_pe == 0) ? num_pes - 1 : my_pe - 1;
    nvshmem_quiet();
    nvshmemx_signal_op(sync_arr + 1, counter, NVSHMEM_SIGNAL_SET, next_rank);
    nvshmemx_signal_op(sync_arr,     counter, NVSHMEM_SIGNAL_SET, prev_rank);
    nvshmem_uint64_wait_until_all(sync_arr, 2, NULL, NVSHMEM_CMP_GE, counter);
}

int main(int argc, char **argv) {
    clock_t global_start = clock();
    double t_nvshmem_prep_start = MPI_Wtime();

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    int* ghost_nys = (int*)malloc(size * sizeof(int));
    int* local_nys = (int*)malloc(size * sizeof(int));
    MPI_Allgather(&local_ny, 1, MPI_INT, local_nys, 1, MPI_INT, MPI_COMM_WORLD);
    for(int i=0; i<size; i++) ghost_nys[i] = local_nys[i] + 2;

    int max_ghost_ny;
    MPI_Allreduce(&ghost_ny, &max_ghost_ny, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    size_t bytes = max_ghost_ny * nx * sizeof(double);
    size_t required_heap = 2 * bytes * 1.1;
    if (!getenv("NVSHMEM_SYMMETRIC_SIZE")) {
        char buf[64];
        sprintf(buf, "%zu", required_heap);
        setenv("NVSHMEM_SYMMETRIC_SIZE", buf, 1);
    }

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    double t_nvshmem_prep_end = MPI_Wtime();

    double t_nvshmem_init_start = MPI_Wtime();
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    double t_nvshmem_init_end = MPI_Wtime();

    nvshmem_barrier_all();
    double t_after_nvshmem_init = MPI_Wtime();

    double *d_a     = (double*)nvshmem_malloc(bytes);
    double *d_a_new = (double*)nvshmem_malloc(bytes);

    uint64_t* sync_arr = (uint64_t*)nvshmem_malloc(2 * sizeof(uint64_t));
    cudaMemset(sync_arr, 0, 2 * sizeof(uint64_t));
    uint64_t sync_counter = 1;

    dim3 block(BLOCK, BLOCK);
    dim3 grid((nx + BLOCK-1)/BLOCK, (ghost_ny + BLOCK-1)/BLOCK);
    init_boundaries<<<grid, block>>>(d_a, d_a_new, nx, ghost_ny);
    cudaDeviceSynchronize();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvshmem_barrier_all();
    double t_calc0 = MPI_Wtime();

    int prev = (rank - 1 + size) % size;
    int next = (rank + 1) % size;
    double error = tol + 1.0;
    int iter = 0;

    // Pour overlap: découpage en deux kernels + overlap
    dim3 grid_inner((nx + BLOCK-1)/BLOCK, (ghost_ny-4 + BLOCK-1)/BLOCK);
    dim3 grid_border((nx + BLOCK-1)/BLOCK, 1);

    while (error > tol && iter < max_iter) {
        // --- 1. Calcul interne en overlap (iy = 2 à ghost_ny-3 inclus)
        jacobi_inner_kernel<<<grid_inner, block, 0, stream>>>(d_a, d_a_new, nx, ghost_ny);

        // --- 2. Pendant ce temps, échange des halos ---
        nvshmem_double_put(
            d_a + IDX(ghost_nys[next]-1, 0, nx),
            d_a + IDX(local_ny, 0, nx),
            nx, next);
        nvshmem_double_put(
            d_a + IDX(0, 0, nx),
            d_a + IDX(1, 0, nx),
            nx, prev);
        nvshmem_fence();

        // --- 3. Synchronisation neighborhood device (bloque jusqu'à halos à jour)
        syncneighborhood_kernel<<<1,1,0,stream>>>(rank, size, sync_arr, sync_counter);
        cudaStreamSynchronize(stream);
        sync_counter++;

        // --- 4. Calcul sur les lignes du bord haut et bas (iy = 1 et iy = ghost_ny-2)
        jacobi_border_kernel<<<grid_border, block, 0, stream>>>(d_a, d_a_new, nx, ghost_ny);
        cudaStreamSynchronize(stream);

        // --- 5. Calcul de l'erreur max (global)
        thrust::device_ptr<double> ptr_new(d_a_new);
        thrust::device_ptr<double> ptr_old(d_a);
        error = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new, ptr_old)),
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new + ghost_ny*nx, ptr_old + ghost_ny*nx)),
            max_abs_diff(),
            0.0,
            thrust::maximum<double>());
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        swap_ptrs(&d_a, &d_a_new);
        iter++;
    }

    nvshmem_barrier_all();
    double t_calc1 = MPI_Wtime();

    nvshmem_barrier_all();
    double t_after_all = MPI_Wtime();

    if (rank == 0) {
        printf("[NVSHMEM-norm_overlap-neighSync+LTO] Converged in %d iterations | Error: %.2e\n", iter, error);
        printf("Temps de calcul Jacobi (seulement boucle)         : %.6fs\n", t_calc1 - t_calc0);
        printf("Temps setup+calcul+affichage (après init SHMEM)   : %.6fs\n", t_after_all - t_after_nvshmem_init);
        printf("Temps d'init avant NVSHMEM                        : %.6fs\n", t_nvshmem_prep_end - t_nvshmem_prep_start);
        printf("Temps d'init NVSHMEM                              : %.6fs\n", t_nvshmem_init_end - t_nvshmem_init_start);
        double global_time = (double)(clock() - global_start) / CLOCKS_PER_SEC;
        printf("Temps total du programme (tout compris)           : %.6fs\n", global_time);
    }

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
    if(sync_arr) nvshmem_free(sync_arr);
    cudaStreamDestroy(stream);
    nvshmem_finalize();
    MPI_Finalize();
    free(ghost_nys);
    free(local_nys);
    return 0;
}
