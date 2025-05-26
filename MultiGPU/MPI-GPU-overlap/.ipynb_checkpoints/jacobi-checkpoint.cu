#include <mpi.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>    // Pour la mesure du temps global

#define IDX(iy, ix, nx) ((iy)*(nx)+(ix))
#define BLOCK 16

// Affichage sur l’host de la grille complète
void printState(const double* a, int nx, int ny) {
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++)
            printf("%.6f ", a[iy * nx + ix]);
        printf("\n");
    }
    printf("\n");
}

// Initialisation des bords : conditions de Dirichlet sur les bords verticaux à 1, reste à 0
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

// Kernel Jacobi qui agit sur une bande verticale [iy_start, iy_end[
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

// Pour la réduction Thrust : max absolu des différences
struct max_abs_diff {
    __host__ __device__
    double operator()(const thrust::tuple<double,double>& t) const {
        return fabs(thrust::get<0>(t) - thrust::get<1>(t));
    }
};

static inline void swap_ptrs(double **p, double **q) {
    double *t=*p; *p=*q; *q=t;
}

int main(int argc, char **argv) {
    // Chrono global pour TOUT le programme
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

    // équilibre même si ny-2 non multiple de size 
    int work_ny = ny - 2; // lignes intérieures à répartir
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
    int ghost_ny = local_ny + 2; // on ajoute deux lignes fantômes

    size_t bytes = ghost_ny * nx * sizeof(double);
    double *d_a, *d_a_new;
    cudaMalloc(&d_a,     bytes);
    cudaMalloc(&d_a_new, bytes);

    // Initialisation sur le GPU
    dim3 block(BLOCK, BLOCK), grid((nx+BLOCK-1)/BLOCK, (ghost_ny+BLOCK-1)/BLOCK);
    init_boundaries<<<grid, block>>>(d_a, d_a_new, nx, ghost_ny);
    cudaDeviceSynchronize();

    // Création de plusieurs streams CUDA pour pouvoir overlaper transferts et calculs
    cudaStream_t stream_halo_top, stream_halo_bot, stream_interior, stream_memcpy;
    cudaStreamCreate(&stream_halo_top);
    cudaStreamCreate(&stream_halo_bot);
    cudaStreamCreate(&stream_interior);
    cudaStreamCreate(&stream_memcpy);

    // Allocation de buffers host pour les halos à envoyer et recevoir
    double *h_halo_top_send = (double*)malloc(nx * sizeof(double));
    double *h_halo_top_recv = (double*)malloc(nx * sizeof(double));
    double *h_halo_bot_send = (double*)malloc(nx * sizeof(double));
    double *h_halo_bot_recv = (double*)malloc(nx * sizeof(double));

    // Chrono scientifique pour le coeur du calcul (MPI+GPU)
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int prev = (rank-1+size)%size, next = (rank+1)%size;
    double error = tol + 1.0;
    int iter = 0;

    thrust::device_ptr<double> ptr_new(d_a);
    thrust::device_ptr<double> ptr_old(d_a_new);
    size_t local_N = ghost_ny * nx;

    while (error > tol && iter < max_iter) {
        MPI_Request reqs[4];

        // On commence par lancer les copies asynchrones des halos device -> host
        cudaMemcpyAsync(h_halo_top_send, d_a + IDX(1,0,nx), nx*sizeof(double), cudaMemcpyDeviceToHost, stream_memcpy);
        cudaMemcpyAsync(h_halo_bot_send, d_a + IDX(local_ny,0,nx), nx*sizeof(double), cudaMemcpyDeviceToHost, stream_memcpy);

        // On synchronise juste ce stream (memcpy), pas les autres !
        cudaStreamSynchronize(stream_memcpy);

        // On lance tout de suite les échanges asynchrones MPI sur les halos
        MPI_Irecv(h_halo_top_recv, nx, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(h_halo_bot_recv, nx, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(h_halo_top_send, nx, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(h_halo_bot_send, nx, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[3]);

        // Pendant que le MPI tourne, on calcule l'intérieur de la grille (hors bords)
        int iy_start_interior = 2;
        int iy_end_interior = local_ny;
        if (iy_end_interior > iy_start_interior) {
            dim3 grid_interior((nx+BLOCK-1)/BLOCK, (iy_end_interior-iy_start_interior+BLOCK-1)/BLOCK);
            jacobi_kernel_slice<<<grid_interior, block, 0, stream_interior>>>(d_a, d_a_new, nx, iy_start_interior, iy_end_interior);
        }

        // On attend que la réception des halos soit finie
        MPI_Wait(&reqs[0], MPI_STATUS_IGNORE); // halo du haut
        MPI_Wait(&reqs[1], MPI_STATUS_IGNORE); // halo du bas

        // On transfère immédiatement les halos reçus host->device, en async aussi
        cudaMemcpyAsync(d_a + IDX(0,0,nx), h_halo_top_recv, nx*sizeof(double), cudaMemcpyHostToDevice, stream_memcpy);
        cudaMemcpyAsync(d_a + IDX(ghost_ny-1,0,nx), h_halo_bot_recv, nx*sizeof(double), cudaMemcpyHostToDevice, stream_memcpy);

        // On attend la fin du transfert avant de lancer les calculs des bandes extrêmes
        cudaStreamSynchronize(stream_memcpy);

        // Calcul des bandes extrêmes (haut et bas) quand les halos sont prêts
        dim3 grid_band((nx+BLOCK-1)/BLOCK, 1);
        jacobi_kernel_slice<<<grid_band, block, 0, stream_halo_top>>>(d_a, d_a_new, nx, 1, 2);
        jacobi_kernel_slice<<<grid_band, block, 0, stream_halo_bot>>>(d_a, d_a_new, nx, local_ny, local_ny+1);

        // On attend la fin de tous les streams CUDA avant la réduction de l'erreur
        cudaStreamSynchronize(stream_interior);
        cudaStreamSynchronize(stream_halo_top);
        cudaStreamSynchronize(stream_halo_bot);

        // On s'assure que les envois MPI sont terminés avant de passer à la suite
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Calcul de l'erreur max (pour convergence), puis MPI_Allreduce
        error = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new, ptr_old)),
            thrust::make_zip_iterator(thrust::make_tuple(ptr_new + local_N, ptr_old + local_N)),
            max_abs_diff(),
            0.0,
            thrust::maximum<double>());
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        swap_ptrs(&d_a, &d_a_new);
        iter++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Affichage des deux chronos (calcul pur et temps total)
    if (!rank) {
        printf("Solveur Jacobi MPI+multiGPU (full overlap memcpyasync) convergé en %d itérations (error = %.3e)\n", iter, error);
        printf("Temps calcul MPI+multiGPU (hors alloc/init): %.6f s\n", t1-t0);
        double global_time = (double)(clock() - global_start) / CLOCKS_PER_SEC;
        printf("Temps total du programme (tout inclus): %.6f s\n", global_time);
    }

    // Rassemblement des résultats pour affichage sur le rang 0 
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
            // On copie les lignes du haut et du bas (périodicité pour affichage)
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
    cudaStreamDestroy(stream_memcpy);

    free(h_halo_top_send); free(h_halo_top_recv);
    free(h_halo_bot_send); free(h_halo_bot_recv);

    MPI_Finalize();
    return 0;
}
