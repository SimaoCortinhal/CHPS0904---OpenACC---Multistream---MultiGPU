#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime> // Pour la mesure du temps total
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

// Le kernel Jacobi sur mon sous-domaine local (slice)
__global__
void jacobi_kernel_slice(const double *a, double *a_new, int nx, int iy_start, int iy_end) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    if (ix < 1 || ix >= nx-1 || iy < iy_start || iy >= iy_end) return;
    int idx = iy * nx + ix;
    a_new[idx] = 0.25 * (
        a[idx - nx] +   // haut
        a[idx + nx] +   // bas
        a[idx - 1 ] +   // gauche
        a[idx + 1 ]     // droite
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
static inline void swap_ptrs(double **p, double **q) {
    double *t=*p; *p=*q; *q=t;
}

int main(int argc, char **argv) {
    // Chronomètre global pour mesurer tout le programme, initialisations incluses
    double t_total0 = MPI_Wtime();
    clock_t global_start = clock(); // Peut servir à mesurer un temps CPU global si besoin

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Attribution d’un GPU par processus MPI
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    cudaSetDevice(rank % num_gpus);

    // Initialisation du communicator NCCL à partir de MPI
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Mesure du temps d'init NCCL
    double t_nccl_init_start = MPI_Wtime();
    ncclComm_t nccl_comm;
    ncclCommInitRank(&nccl_comm, size, id, rank);
    double t_nccl_init_end = MPI_Wtime();
    if (rank == 0)
        printf("NCCL Comm Init: %.3f s\n", t_nccl_init_end-t_nccl_init_start);

    // Chronomètre juste après NCCL init (début "setup CUDA et allocations")
    double t_after_nccl = MPI_Wtime();

    if (argc != 6) {
        if (!rank)
            fprintf(stderr,"Usage: %s nx ny max_iter tol print_flag\n", argv[0]);
        ncclCommDestroy(nccl_comm);
        MPI_Finalize();
        return 1;
    }
    int nx         = atoi(argv[1]);
    int ny         = atoi(argv[2]);
    int max_iter   = atoi(argv[3]);
    double tol     = atof(argv[4]);
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

    // Allocations sur le GPU
    size_t bytes = ghost_ny * nx * sizeof(double);
    double *d_a, *d_a_new;
    cudaMalloc(&d_a,     bytes);
    cudaMalloc(&d_a_new, bytes);

    // Initialisation des bords sur le device
    dim3 block(BLOCK, BLOCK), grid((nx+BLOCK-1)/BLOCK, (ghost_ny+BLOCK-1)/BLOCK);
    init_boundaries<<<grid, block>>>(d_a, d_a_new, nx, ghost_ny);
    cudaDeviceSynchronize();

    // Création de plusieurs streams CUDA (overlap comm/calcul)
    cudaStream_t stream_halo_top, stream_halo_bot, stream_interior;
    cudaStreamCreate(&stream_halo_top);
    cudaStreamCreate(&stream_halo_bot);
    cudaStreamCreate(&stream_interior);

    // Chrono pour le temps de calcul pur (juste la boucle Jacobi)
    MPI_Barrier(MPI_COMM_WORLD);
    double t_calc0 = MPI_Wtime();

    int prev = (rank-1+size)%size, next = (rank+1)%size;
    double error = tol + 1.0;
    int iter = 0;

    thrust::device_ptr<double> ptr_new(d_a);
    thrust::device_ptr<double> ptr_old(d_a_new);
    size_t local_N = ghost_ny * nx;

    // --- BOUCLE PRINCIPALE JACOBI ---
    while (error > tol && iter < max_iter) {
        // 1) NCCL comm overlap (échange des halos haut/bas)
        ncclGroupStart();
        ncclRecv(d_a + IDX(0,0,nx), nx, ncclDouble, prev, nccl_comm, stream_halo_top); // haut
        ncclSend(d_a + IDX(1,0,nx), nx, ncclDouble, prev, nccl_comm, stream_halo_top);
        ncclRecv(d_a + IDX(ghost_ny-1,0,nx), nx, ncclDouble, next, nccl_comm, stream_halo_bot); // bas
        ncclSend(d_a + IDX(local_ny,0,nx), nx, ncclDouble, next, nccl_comm, stream_halo_bot);
        ncclGroupEnd();

        // 2) Calcul extrémités EN OVERLAP avec comm (un kernel par stream)
        dim3 grid_band((nx+BLOCK-1)/BLOCK, 1);
        jacobi_kernel_slice<<<grid_band, block, 0, stream_halo_top>>>(d_a, d_a_new, nx, 1, 2);
        jacobi_kernel_slice<<<grid_band, block, 0, stream_halo_bot>>>(d_a, d_a_new, nx, local_ny, local_ny+1);

        // 3) Calcul intérieur (indépendant des halos) sur un autre stream
        int iy_start_interior = 2;
        int iy_end_interior = local_ny;
        if (iy_end_interior > iy_start_interior) {
            dim3 grid_interior((nx+BLOCK-1)/BLOCK, (iy_end_interior-iy_start_interior+BLOCK-1)/BLOCK);
            jacobi_kernel_slice<<<grid_interior, block, 0, stream_interior>>>(d_a, d_a_new, nx, iy_start_interior, iy_end_interior);
        }

        // 4) Synchronisation fine : on attend chaque stream
        cudaStreamSynchronize(stream_halo_top);
        cudaStreamSynchronize(stream_halo_bot);
        cudaStreamSynchronize(stream_interior);

        swap_ptrs(&d_a, &d_a_new);

        // Calcul erreur max locale puis réduction MPI
        error = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new, ptr_old)),
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new + local_N, ptr_old + local_N)),
            max_abs_diff(),
            0.0,
            thrust::maximum<double>());
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        iter++;
    }
    // --- FIN BOUCLE JACOBI ---

    MPI_Barrier(MPI_COMM_WORLD);
    double t_calc1 = MPI_Wtime();

    // Chrono total (fin)
    double t_total1 = MPI_Wtime();

    // Affichage sur le rang 0 des trois chronos :
    if (!rank) {
        double t_total      = t_total1 - t_total0;
        double t_after_init = t_total1 - t_after_nccl;
        double t_calcul     = t_calc1 - t_calc0;
        double t_cpu        = (double)(clock() - global_start) / CLOCKS_PER_SEC;
        printf("Solveur Jacobi NCCL + overlap convergé en %d itérations (error = %.3e)\n", iter, error);
        printf("1. Temps total du programme (MPI_Init → fin)                 : %.6f s\n", t_total);
        printf("2. Temps après init NCCL (juste avant alloc/init CUDA)       : %.6f s\n", t_after_init);
        printf("3. Temps calcul NCCL + overlap (boucle Jacobi uniquement)    : %.6f s\n", t_calcul);
        // printf("Temps total (CPU clock, juste indicatif): %.6f s\n", t_cpu);
    }

    // Réassemblage et affichage comme avant (si demandé)
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
    cudaStreamDestroy(stream_halo_top);
    cudaStreamDestroy(stream_halo_bot);
    cudaStreamDestroy(stream_interior);
    ncclCommDestroy(nccl_comm);
    MPI_Finalize();
    return 0;
}
