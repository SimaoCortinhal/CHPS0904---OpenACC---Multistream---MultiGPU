#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime> // Pour le chrono CPU global
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>

#define IDX(iy, ix, nx) ((iy)*(nx)+(ix))
#define BLOCK 16

// Affichage d'une grille 2D sur le host
void printState(const double* a, int nx, int ny) {
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++)
            printf("%.6f ", a[iy * nx + ix]);
        printf("\n");
    }
    printf("\n");
}

// Initialisation : bords verticaux à 1, intérieur à 0
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

// Kernel Jacobi pour une bande verticale de lignes (iy_start à iy_end non inclus)
__global__
void jacobi_kernel_slice(const double *a, double *a_new, int nx, int iy_start, int iy_end) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    if (ix < 1 || ix >= nx-1 || iy < iy_start || iy >= iy_end) return;
    int idx = iy * nx + ix;
    a_new[idx] = 0.25 * (
        a[idx - nx] +
        a[idx + nx] +
        a[idx - 1 ] +
        a[idx + 1 ]
    );
}

// Calcul du max des différences absolues entre deux tableaux (pour l'arrêt du Jacobi)
struct max_abs_diff {
    __host__ __device__
    double operator()(const thrust::tuple<double,double>& t) const {
        return fabs(thrust::get<0>(t) - thrust::get<1>(t));
    }
};

// Fonction utilitaire pour swap les pointeurs
static inline void swap_ptrs(double **p, double **q) {
    double *t=*p; *p=*q; *q=t;
}

int main(int argc, char **argv) {
    // Chrono global pour mesurer TOUT le temps du programme
    clock_t global_start = clock();
    double t_total0 = MPI_Wtime();

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Attribution d'un GPU à chaque processus MPI via le local_rank (multi-GPU, multi-nœud)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    int local_rank = 0;
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
    }
    cudaSetDevice(local_rank);

    // Initialisation du communicator NCCL à partir de MPI, mesure du temps NCCL init
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    double t_nccl_init_start = MPI_Wtime();
    ncclComm_t nccl_comm;
    ncclCommInitRank(&nccl_comm, size, id, rank);
    double t_nccl_init_end = MPI_Wtime();
    if (rank == 0)
        printf("[NCCL] Temps init NCCL : %.6fs\n", t_nccl_init_end-t_nccl_init_start);

    // Chrono juste après l'init NCCL (pour mesurer les phases d'alloc/init CUDA)
    double t_after_nccl = MPI_Wtime();

    // Lecture des arguments
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

    // Découpage du domaine Y : chaque processus gère un "bloc" de lignes, équilibré
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
    int ghost_ny = local_ny + 2; // +2 halos (haut/bas)

    // Allocation mémoire device pour les deux grilles (ping-pong Jacobi)
    size_t bytes = ghost_ny * nx * sizeof(double);
    double *d_a, *d_a_new;
    cudaMalloc(&d_a,     bytes);
    cudaMalloc(&d_a_new, bytes);

    // Initialisation des bords sur le GPU (kernel)
    dim3 block(BLOCK, BLOCK), grid((nx+BLOCK-1)/BLOCK, (ghost_ny+BLOCK-1)/BLOCK);
    init_boundaries<<<grid, block>>>(d_a, d_a_new, nx, ghost_ny);
    cudaDeviceSynchronize();

    // Un seul stream CUDA pour la version avec CUDA Graphs (plus simple/rapide ici)
    cudaStream_t s;
    cudaStreamCreate(&s);

    // CUDA Graphs : on capture deux graphes (parité) pour accélérer la boucle Jacobi
    cudaGraph_t       graph[2];
    cudaGraphExec_t   graphExec[2];

    int prev = (rank-1+size)%size, next = (rank+1)%size;
    int iy_start_interior = 2;
    int iy_end_interior = local_ny;

    for (int g=0; g<2; g++) {
        if (g==1) swap_ptrs(&d_a, &d_a_new);

        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);

        // Calcul des bandes haut/bas (après échanges de halos)
        dim3 grid_band((nx+BLOCK-1)/BLOCK, 1);
        jacobi_kernel_slice<<<grid_band, block, 0, s>>>(d_a, d_a_new, nx, 1, 2);
        jacobi_kernel_slice<<<grid_band, block, 0, s>>>(d_a, d_a_new, nx, local_ny, local_ny+1);

        // Calcul de l'intérieur (hors bandes)
        if (iy_end_interior > iy_start_interior) {
            dim3 grid_interior((nx+BLOCK-1)/BLOCK, (iy_end_interior-iy_start_interior+BLOCK-1)/BLOCK);
            jacobi_kernel_slice<<<grid_interior, block, 0, s>>>(d_a, d_a_new, nx, iy_start_interior, iy_end_interior);
        }

        // NCCL : échanges de halos haut/bas en groupe
        ncclGroupStart();
        ncclRecv(d_a_new + IDX(0,0,nx), nx, ncclDouble, prev, nccl_comm, s);
        ncclSend(d_a_new + IDX(1,0,nx), nx, ncclDouble, prev, nccl_comm, s);
        ncclRecv(d_a_new + IDX(ghost_ny-1,0,nx), nx, ncclDouble, next, nccl_comm, s);
        ncclSend(d_a_new + IDX(local_ny,0,nx), nx, ncclDouble, next, nccl_comm, s);
        ncclGroupEnd();

        cudaStreamEndCapture(s, &graph[g]);
        cudaGraphInstantiate(&graphExec[g], graph[g], NULL, NULL, 0);

        if (g==1) swap_ptrs(&d_a, &d_a_new);
    }

    // Début chrono calcul pur (boucle Jacobi)
    MPI_Barrier(MPI_COMM_WORLD);
    double t_calc0 = MPI_Wtime();

    int iter = 0;
    double error = tol + 1.0;
    thrust::device_ptr<double> ptr_new(d_a_new);
    thrust::device_ptr<double> ptr_old(d_a);
    size_t local_N = ghost_ny * nx;

    // Boucle principale Jacobi avec lancement de CUDA Graphs (alternance parité)
    while (error > tol && iter < max_iter) {
        int idx = iter & 1;
        cudaGraphLaunch(graphExec[idx], s);
        cudaStreamSynchronize(s);

        // Toutes les 10 itérations, calcul de la norme d'erreur (critère de convergence)
        if (iter % 10 == 0) {
            error = thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(ptr_new, ptr_old)),
                thrust::make_zip_iterator(thrust::make_tuple(ptr_new + local_N, ptr_old + local_N)),
                max_abs_diff(),
                0.0,
                thrust::maximum<double>());
            MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }
        iter++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t_calc1 = MPI_Wtime();

    // Chrono global (fin)
    double t_total1 = MPI_Wtime();

    // Affichage des chronos sur le rang 0
    if (!rank) {
        double t_total      = t_total1 - t_total0;
        double t_after_init = t_total1 - t_after_nccl;
        double t_calcul     = t_calc1 - t_calc0;
        double t_cpu        = (double)(clock() - global_start) / CLOCKS_PER_SEC;
        printf("[NCCL+Graphs] Jacobi convergé en %d itérations (error = %.3e)\n", iter, error);
        printf("1. Temps total du programme (MPI_Init → fin)                 : %.6fs\n", t_total);
        printf("2. Temps après init NCCL (juste avant alloc/init CUDA)       : %.6fs\n", t_after_init);
        printf("3. Temps calcul Jacobi (boucle CUDA Graphs uniquement)       : %.6fs\n", t_calcul);
        // printf("Temps CPU clock (juste pour info)                            : %.6fs\n", t_cpu);
    }

    // Réassemblage et affichage de la grille finale si demandé
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
    cudaGraphExecDestroy(graphExec[0]);
    cudaGraphExecDestroy(graphExec[1]);
    ncclCommDestroy(nccl_comm);
    cudaStreamDestroy(s);
    MPI_Finalize();
    return 0;
}
