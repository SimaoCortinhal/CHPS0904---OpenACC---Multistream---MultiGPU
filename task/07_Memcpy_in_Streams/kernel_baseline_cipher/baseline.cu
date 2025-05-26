#include <cstdint>
#include <iostream>
#include "helpers.cuh"
#include "encryption.cuh"

// Host function.
void encrypt_cpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters, bool parallel=true) {

    // Use OpenMP to use all available CPU cores.
    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        // Permute each data entry the number of iterations and then write result to data.
        data[entry] = permute64(entry, num_iters);
}

// Device function.
__global__ 
void decrypt_gpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters) {

    const uint64_t thrdID = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = blockDim.x * gridDim.x;

    // Utilize grid-stride loop for arbitrary data sizes.
    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        // Unpermute each data entry the number of iterations then write result to data.
        data[entry] = unpermute64(data[entry], num_iters);
}

// Host function.
bool check_result_cpu(uint64_t * data, uint64_t num_entries, bool parallel=true) {

    uint64_t counter = 0;

    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        // Because we created initial data values by ranging from 0 to N-1,
        // and because encrypting and decrypting is symmetrical,
        // then each data entry should be equal to `entry`.
        counter += data[entry] == entry;

    // True if all values have been correctly decrypted.
    return counter == num_entries;
}

int main (int argc, char * argv[]) {
    Timer tot;
    tot.start();
    // This file will be used to cache encryption results
    // so we don't have to wait on the CPU every time.
    const char * encrypted_file = "/dli/task/encrypted";

    // Timer instance to be used for sections of the application.
    Timer timer;
    
    // Timer instance to be used for total time on the GPU(s).
    Timer overall;

    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters = 1UL << 10;
    
    // Use all available CPUs in parallel for host calculations.
    const bool openmp = true;

    timer.start();
    uint64_t * data_cpu, * data_gpu;
    // cudaMallocHost will be discussed at length later in the course.
    cudaMallocHost(&data_cpu, sizeof(uint64_t) * num_entries);
    cudaMalloc(&data_gpu, sizeof(uint64_t) * num_entries);
    timer.stop("allocate memory");
    check_last_error();

    timer.start();
    // If encryption cache file does not exist...
    if (!encrypted_file_exists(encrypted_file)) {
        // ...encrypt data in parallel on CPU...
        encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
        // ...and make encryption cache file for later.
        write_encrypted_to_file(encrypted_file, data_cpu, sizeof(uint64_t) * num_entries);
    } else {
        // Use encryption cache file if it exists.
        read_encrypted_from_file(encrypted_file, data_cpu, sizeof(uint64_t) * num_entries);
    }
    timer.stop("encrypt data on CPU");

    // Créez un stream non par défaut pour les transferts asynchrones.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Begin timing for total time on GPU(s).
    overall.start();
    
    timer.start();
    // Transfert asynchrone de la mémoire du CPU vers le GPU dans le stream non par défaut.
    cudaMemcpyAsync(data_gpu, data_cpu, sizeof(uint64_t) * num_entries, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    timer.stop("copy data from CPU to GPU");
    check_last_error();

    timer.start();
    // Décryptage sur le GPU (kernel lancé dans le stream par défaut).
    decrypt_gpu<<<80 * 32, 64>>>(data_gpu, num_entries, num_iters);
    timer.stop("decrypt data on GPU");
    check_last_error();

    timer.start();
    // Transfert asynchrone de la mémoire du GPU vers le CPU dans le stream non par défaut.
    cudaMemcpyAsync(data_cpu, data_gpu, sizeof(uint64_t) * num_entries, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    timer.stop("copy data from GPU to CPU");
    // Stop timing total time on GPU(s).
    overall.stop("total time on GPU");
    check_last_error();

    // Détruire le stream car il n'est plus nécessaire.
    cudaStreamDestroy(stream);

    timer.start();
    // Vérification des résultats sur le CPU.
    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << (success ? "passed" : "failed")
              << std::endl;
    timer.stop("checking result on CPU");

    timer.start();
    // Libération de la mémoire.
    cudaFreeHost(data_cpu);
    cudaFree(data_gpu);
    timer.stop("free memory");
    tot.stop("Temps total");
    check_last_error();
}
