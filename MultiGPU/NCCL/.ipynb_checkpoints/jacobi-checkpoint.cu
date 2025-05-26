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
static inline void swap_ptrs(double **p, double **q) {
    double *t=*p; *p=*q; *q=t;
}

int main(int argc, char **argv) {
    // 1. Chronomètre global pour mesurer tout le programme, initialisations incluses
    double t_total0 = MPI_Wtime();
    clock_t t_total0_cpu = clock();

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    cudaSetDevice(rank % num_gpus);

    // Initialisation de NCCL (je récupère le communicator via MPI)
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    ncclCommInitRank(&nccl_comm, size, id, rank);

    // 2. Chronomètre "après init NCCL" (mais AVANT malloc/init mémoire)
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

    size_t bytes = ghost_ny * nx * sizeof(double);
    double *d_a, *d_a_new;
    cudaMalloc(&d_a,     bytes);
    cudaMalloc(&d_a_new, bytes);

    // J'initialise les bords sur le device (tous les bords verticaux à 1, reste à 0)
    dim3 block(BLOCK, BLOCK), grid((nx+BLOCK-1)/BLOCK, (ghost_ny+BLOCK-1)/BLOCK);
    init_boundaries<<<grid, block>>>(d_a, d_a_new, nx, ghost_ny);
    cudaDeviceSynchronize();

    // 3. Chronomètre scientifique pour ne mesurer que le cœur du calcul
    MPI_Barrier(MPI_COMM_WORLD);
    double t_calc0 = MPI_Wtime();

    int prev = (rank-1+size)%size, next = (rank+1)%size;
    double error = tol + 1.0;
    int iter = 0;

    thrust::device_ptr<double> ptr_new(d_a);
    thrust::device_ptr<double> ptr_old(d_a_new);
    size_t local_N = ghost_ny * nx;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    while (error > tol && iter < max_iter) {
        // J'échange les halos avec NCCL (send/recv), un pour chaque voisin
        ncclGroupStart();
        ncclRecv(d_a + IDX(0,0,nx), nx, ncclDouble, prev, nccl_comm, stream);
        ncclSend(d_a + IDX(1,0,nx), nx, ncclDouble, prev, nccl_comm, stream);
        ncclRecv(d_a + IDX(ghost_ny-1,0,nx), nx, ncclDouble, next, nccl_comm, stream);
        ncclSend(d_a + IDX(local_ny,0,nx), nx, ncclDouble, next, nccl_comm, stream);
        ncclGroupEnd();
        cudaStreamSynchronize(stream);

        // Calcul Jacobi sur le sous-domaine local
        jacobi_kernel<<<grid,block,0,stream>>>(d_a, d_a_new, nx, ghost_ny);
        cudaStreamSynchronize(stream);

        swap_ptrs(&d_a,&d_a_new);

        // Calcul de l'erreur max locale (avec thrust) puis réduction MPI pour la convergence globale
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
    double t_calc1 = MPI_Wtime();

    // Affichage des trois chronos (global, pré-calcul, calcul pur)
    if (!rank) {
        double t_total1 = MPI_Wtime();
        double total_program = t_total1 - t_total0;
        double after_nccl    = t_total1 - t_after_nccl;
        double calcul        = t_calc1 - t_calc0;
        double global_time_cpu = (double)(clock() - t_total0_cpu) / CLOCKS_PER_SEC;

        printf("Solveur Jacobi NCCL convergé en %d itérations (error = %.3e)\n", iter, error);
        printf("1. Temps total du programme (MPI_Init → fin)                : %.6f s\n", total_program);
        printf("2. Temps après init NCCL (juste avant alloc/init CUDA)      : %.6f s\n", after_nccl);
        printf("3. Temps NCCL (calcul pur boucle Jacobi)                    : %.6f s\n", calcul);
        // printf("Temps total (CPU clock, juste indicatif): %.6f s\n", global_time_cpu);
    }

    // Je rassemble tout et j'affiche le résultat final (comme avant)
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
    cudaStreamDestroy(stream);
    ncclCommDestroy(nccl_comm);
    MPI_Finalize();
    return 0;
}
