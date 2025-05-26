#include <cstdint>
#include <iostream>
#include <algorithm> // Pour std::min
#include "helpers.cuh"
#include "encryption.cuh"

// Fonction host pour encrypter les données.
void encrypt_cpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters, bool parallel=true) {
    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        data[entry] = permute64(entry, num_iters);
}

// Kernel de décryptage sur le GPU.
__global__ 
void decrypt_gpu(uint64_t * data, uint64_t num_entries, uint64_t num_iters) {
    const uint64_t thrdID = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = blockDim.x * gridDim.x;

    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        data[entry] = unpermute64(data[entry], num_iters);
}

// Fonction host pour vérifier les résultats.
bool check_result_cpu(uint64_t * data, uint64_t num_entries, bool parallel=true) {
    uint64_t counter = 0;
    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        counter += (data[entry] == entry);
    return counter == num_entries;
}
int main(int argc, char* argv[]) {
    Timer tot; tot.start();

    const char* encrypted_file = "/dli/task/encrypted";
    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters   = 1UL << 10;
    const bool openmp = true;

    uint64_t* data_cpu = nullptr;
    cudaMallocHost(&data_cpu, sizeof(uint64_t) * num_entries);
    check_last_error();


    if (!encrypted_file_exists(encrypted_file)) {
        encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
        write_encrypted_to_file(encrypted_file, data_cpu,
                                sizeof(uint64_t) * num_entries);
    } else {
        read_encrypted_from_file(encrypted_file, data_cpu,
                                 sizeof(uint64_t) * num_entries);
    }

    // Nombre de GPUs et découpage des données
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus <= 0) {
        std::cerr << "Aucun GPU disponible." << std::endl;
        return 1;
    }
    std::cout << "Nombre de GPUs disponibles : " << num_gpus << std::endl;

    // taille de chunk par GPU (division arrondie vers le haut)
    uint64_t chunk_size = sdiv(num_entries, num_gpus);

    // stockage des pointeurs device et des streams
    std::vector<uint64_t*> data_gpu(num_gpus, nullptr);
    std::vector<cudaStream_t> streams(num_gpus);

    // tableaux pour les offsets et tailles réelles
    std::vector<uint64_t> lower(num_gpus), width(num_gpus);
    std::vector<size_t> bytes(num_gpus);

    // Init par GPU
    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);

        // compute bounds
        lower[g] = chunk_size * g;
        uint64_t upper = std::min(lower[g] + chunk_size, num_entries);
        width[g] = upper - lower[g];
        bytes[g] = width[g] * sizeof(uint64_t);

        // alloc device
        cudaMalloc(&data_gpu[g], bytes[g]);
        check_last_error();

        // create one stream per GPU
        cudaStreamCreate(&streams[g]);
        check_last_error();
    }

    Timer kernel_timer; kernel_timer.start();

    //  copies Host→Device
    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        cudaMemcpyAsync(
          data_gpu[g],
          data_cpu + lower[g],
          bytes[g],
          cudaMemcpyHostToDevice,
          streams[g]
        );
    }

    // lancement des kernels
    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);

        // calcul dynamique de blocks & threads
        int threadsPerBlock = 256;
        int fullBlocks = (width[g] + threadsPerBlock - 1) / threadsPerBlock;
        // Optionnel : limiter les blocs si on veut laisser du SM libre
        int smCount;
        cudaDeviceGetAttribute(&smCount,
          cudaDevAttrMultiProcessorCount, g);
        int maxBlocks = smCount * 2;  // 2 blocs par SM, par exemple
        int blocks = std::min(fullBlocks, maxBlocks);

        decrypt_gpu
          <<< blocks, threadsPerBlock, 0, streams[g] >>>(
            data_gpu[g],
            width[g],
            num_iters
          );
    }

    // copies Device→Host
    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        cudaMemcpyAsync(
          data_cpu + lower[g],
          data_gpu[g],
          bytes[g],
          cudaMemcpyDeviceToHost,
          streams[g]
        );
    }

    // Synchronisation de tous les streams
    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        cudaStreamSynchronize(streams[g]);
        check_last_error();
    }
    kernel_timer.stop("Temps de décryptage kernel sur GPUs");

    // Vérification et cleanup
    bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " << (success ? "passed" : "failed") << std::endl;

    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        cudaStreamDestroy(streams[g]);
        cudaFree(data_gpu[g]);
    }
    cudaFreeHost(data_cpu);
    check_last_error();

    tot.stop("Temps total");
    return 0;
}