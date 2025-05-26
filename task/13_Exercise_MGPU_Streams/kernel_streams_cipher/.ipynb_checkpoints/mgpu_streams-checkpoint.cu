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

    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters   = 1UL << 10;
    const bool openmp = true;
    const char* encrypted_file = "/dli/task/encrypted";

    uint64_t* data_cpu = nullptr;
    cudaMallocHost(&data_cpu, num_entries * sizeof(uint64_t));
    if (!encrypted_file_exists(encrypted_file)) {
        encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
        write_encrypted_to_file(encrypted_file, data_cpu,
                                num_entries * sizeof(uint64_t));
    } else {
        read_encrypted_from_file(encrypted_file, data_cpu,
                                 num_entries * sizeof(uint64_t));
    }


    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus <= 0) { std::cerr<<"Aucun GPU\n"; return 1; }
    std::cout<<"GPUs disponibles: "<<num_gpus<<"\n";

    // Paramètres de découpage
    const uint64_t num_streams = 32;  // streams par GPU
    uint64_t per_gpu = sdiv(num_entries, num_gpus);
    uint64_t per_stream = sdiv(per_gpu, num_streams);

    // stocker pointeurs device et streams
    std::vector<std::vector<cudaStream_t>> streams(num_gpus,
                                                   std::vector<cudaStream_t>(num_streams));
    std::vector<uint64_t*> data_gpu(num_gpus, nullptr);

    // Allocation & création de streams
    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        uint64_t slice = std::min(per_gpu, num_entries - g*per_gpu);
        cudaMalloc(&data_gpu[g], slice * sizeof(uint64_t));

        for (int s = 0; s < (int)num_streams; ++s) {
            cudaStreamCreate(&streams[g][s]);
        }
    }

    Timer kt; kt.start();

    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);

        // copies Host→Device sur tous les streams
        for (int s = 0; s < (int)num_streams; ++s) {
            uint64_t global_off = uint64_t(g)*per_gpu + uint64_t(s)*per_stream;
            if (global_off >= num_entries) break;
            uint64_t width = std::min(per_stream, num_entries - global_off);
            size_t bytes = width * sizeof(uint64_t);

            cudaMemcpyAsync(
              data_gpu[g] + uint64_t(s)*per_stream,
              data_cpu  +  global_off,
              bytes,
              cudaMemcpyHostToDevice,
              streams[g][s]
            );
        }

        // lancement des kernels sur tous les streams
        int smCount;
        cudaDeviceGetAttribute(&smCount,
            cudaDevAttrMultiProcessorCount, g);

        for (int s = 0; s < (int)num_streams; ++s) {
            uint64_t global_off = uint64_t(g)*per_gpu + uint64_t(s)*per_stream;
            if (global_off >= num_entries) break;
            uint64_t width = std::min(per_stream, num_entries - global_off);

            // threads/blocs
            int tpB = 256;
            int fullB = int((width + tpB - 1)/tpB);
            int maxB = smCount * 2;            // 2 blocs par SM
            int blocks= std::min(fullB, maxB);

            decrypt_gpu
              <<<blocks, tpB, 0, streams[g][s]>>>(
                data_gpu[g] + uint64_t(s)*per_stream,
                width, num_iters
            );
        }

        // copies Device→Host sur **tous** les streams
        for (int s = 0; s < (int)num_streams; ++s) {
            uint64_t global_off = uint64_t(g)*per_gpu + uint64_t(s)*per_stream;
            if (global_off >= num_entries) break;
            uint64_t width = std::min(per_stream, num_entries - global_off);
            size_t bytes = width * sizeof(uint64_t);

            cudaMemcpyAsync(
              data_cpu  +  global_off,
              data_gpu[g] + uint64_t(s)*per_stream,
              bytes,
              cudaMemcpyDeviceToHost,
              streams[g][s]
            );
        }
    }

    // Synchronisation de tous les streams sur tous les GPUs
    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        for (auto& st : streams[g])
            cudaStreamSynchronize(st);
    }
    kt.stop("Temps H2D+compute+D2H");

    // Check & cleanup
    bool ok = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout<<"STATUS: "<<(ok?"PASSED":"FAILED")<<"\n";

    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        for (auto& st : streams[g]) cudaStreamDestroy(st);
        cudaFree(data_gpu[g]);
    }
    cudaFreeHost(data_cpu);
    tot.stop("Temps total");
    return 0;
}