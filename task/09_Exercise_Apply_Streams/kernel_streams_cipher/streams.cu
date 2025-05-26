#include <cstdint>
#include <iostream>
#include <algorithm>    // pour std::min
#include "helpers.cuh"
#include "encryption.cuh"

// On suppose que sdiv est défini dans helpers.cuh et réalise une division arrondie vers le haut

void encrypt_cpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters, bool parallel=true) {

    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        data[entry] = permute64(entry, num_iters);
}

__global__ 
void decrypt_gpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters) {

    const uint64_t thrdID = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = blockDim.x * gridDim.x;

    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        data[entry] = unpermute64(data[entry], num_iters);
}

bool check_result_cpu(uint64_t * data, uint64_t num_entries,
                      bool parallel=true) {

    uint64_t counter = 0;

    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        counter += (data[entry] == entry);

    return counter == num_entries;
}

int main (int argc, char * argv[]) {
    Timer tot;
    tot.start();
    cudaDeviceProp p;
cudaGetDeviceProperties(&p, 0);
std::cout
  << "concurrentKernels: " << p.concurrentKernels
  << ", asyncEngineCount: "  << p.asyncEngineCount
  << "\n";

    const char * encrypted_file = "/dli/task/encrypted";

    Timer timer;
    

    const uint64_t num_entries = 1UL << 26;  // Nombre total d'entrées
    const uint64_t num_iters   = 1UL << 10;
    const bool openmp = true;
    
    uint64_t * data_cpu = nullptr;
    uint64_t * data_gpu = nullptr;
    cudaMallocHost(&data_cpu, sizeof(uint64_t) * num_entries);
    cudaMalloc(&data_gpu, sizeof(uint64_t) * num_entries);
    check_last_error();

    if (!encrypted_file_exists(encrypted_file)) {
        encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
        write_encrypted_to_file(encrypted_file, data_cpu, sizeof(uint64_t) * num_entries);
    } else {
        read_encrypted_from_file(encrypted_file, data_cpu, sizeof(uint64_t) * num_entries);
    }

    timer.start();
    // --- Mise en place des streams pour le copy/compute overlap ---
    const int num_streams = 32;   // Vous pouvez expérimenter avec ce nombre
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calcul de la taille de chunk pour chaque stream (division arrondie vers le haut)
    const uint64_t chunk = sdiv(num_entries, num_streams);

    
    // Pour chaque stream, lancer les transferts asynchrones et le calcul
    for (int i = 0; i < num_streams; i++) {
        // Calculer l’offset et la taille (le dernier chunk peut être plus petit)
        uint64_t offset = i * chunk;
        uint64_t current_chunk = std::min(chunk, num_entries - offset);
        size_t bytes = current_chunk * sizeof(uint64_t);

        // Transfert asynchrone du host vers le device pour ce segment
        cudaMemcpyAsync(data_gpu + offset, data_cpu + offset, bytes,
                        cudaMemcpyHostToDevice, streams[i]);

        // Lancement du noyau sur le segment correspondant
        int threadsPerBlock = 32;
        int smCount;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
        // on veut 2 blocs actif·ve·s par SM, quitte à n’avoir que chunk partiel traité en un appel
        int maxBlocks = smCount * 2;
        int fullBlocks = (current_chunk + threadsPerBlock - 1) / threadsPerBlock;
        int blocks    = std::min(maxBlocks, fullBlocks);
        decrypt_gpu<<<blocks, threadsPerBlock, 0, streams[i]>>>(data_gpu + offset,
                                                                 current_chunk,
                                                                 num_iters);

        // Transfert asynchrone du device vers le host pour récupérer le résultat
        cudaMemcpyAsync(data_cpu + offset, data_gpu + offset, bytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchroniser tous les streams avant de poursuivre
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    timer.stop("total time on GPU");
    check_last_error();

    // Vérification du résultat sur le CPU
    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " << (success ? "passed" : "failed") << std::endl;

    // Libération des streams et de la mémoire
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(data_cpu);
    cudaFree(data_gpu);
    check_last_error();
    tot.stop("Temps total");

    return 0;
}
