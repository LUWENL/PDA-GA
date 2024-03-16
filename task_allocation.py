import time
from target_track_gym.models.genetic_plain import *
from target_track_gym.models.genetic_kernel import *
from target_track_gym.metadata import METADATA


def ga_plain(N_satellite, N_target, sat_samplings, sat_attitudes,
             sat_vectors, sat_occultated, tar_prioritys, tar_vectors):
    popSize = METADATA['popSize']
    chrom_size = N_satellite
    eliteSize = METADATA['eliteSize']
    crossoverRate = METADATA['crossoverRate']
    mutationRate = METADATA['mutationRate']
    num_generations = METADATA['generations']
    np.random.seed(METADATA['seed'])

    chromosomes = np.zeros(shape=(popSize, chrom_size), dtype=np.int32)
    fitnesses = np.zeros(shape=(popSize, 1), dtype=np.float64)
    tmp_fitnesses = np.zeros(shape=(popSize, 1), dtype=np.float64)

    start = time.perf_counter()


    for i in range(popSize):
        for j in range(chrom_size):
            chromosomes[i][j] = random.randint(0, N_target - 1)



    # Genetic Algorithm on CPU
    for i in range(num_generations + 1):

        eval_genomes_plain(chromosomes, fitnesses, popSize, N_satellite, N_target, sat_samplings,
                           sat_attitudes, sat_vectors, sat_occultated, tar_prioritys, tar_vectors)


        if i < num_generations:
            crossoverRate, mutationRate = adaptive_mechanism_plain(crossoverRate, mutationRate)

            next_generation_plain(chromosomes, fitnesses, tmp_fitnesses, N_target, sat_samplings, sat_attitudes,
                                  sat_vectors, sat_occultated, tar_prioritys, tar_vectors, eliteSize,
                                  crossoverRate, mutationRate)

            fitnesses = np.zeros(shape=[popSize, 1], dtype=np.float64)

    end = time.perf_counter()

    show_fitness_pairs = []
    for i in range(len(chromosomes)):
        show_fitness_pairs.append([chromosomes[i], fitnesses[i]])

    fitnesses = list(reversed(sorted(fitnesses)))  # fitnesses now in descending order
    show_sorted_pairs = list(reversed(sorted(show_fitness_pairs, key=lambda x: x[1])))
    best_allocation = show_sorted_pairs[0][0]
    best_fitness = show_sorted_pairs[0][1]

    return best_allocation, best_fitness, end - start


def ga_kernel(N_satellite, N_target, sat_samplings, sat_attitudes,
              sat_vectors, sat_occultated, tar_prioritys, tar_vectors):
    popSize = METADATA['popSize']
    eliteSize = METADATA['eliteSize']
    chrom_size = N_satellite
    crossoverRate = METADATA['crossoverRate']
    mutationRate = METADATA['mutationRate']
    num_generations = METADATA['generations']

    #  input array
    cuda_chromosomes = cuda.to_device(np.zeros(shape=(popSize, chrom_size), dtype=np.int32))
    cuda_fitnesses = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_sorted_chromosomes = cuda.to_device(np.zeros([popSize, chrom_size], dtype=np.int32))
    cuda_sorted_fitnesses = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_tmp_fitnesses = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))  # only for mutation
    cuda_fitnessTotal = cuda.to_device(np.zeros(shape=1, dtype=np.float64))
    cuda_rouletteWheel = cuda.to_device(np.zeros(shape=popSize, dtype=np.float64))
    cuda_popSize = cuda.to_device(np.array([popSize], dtype=np.int32))
    cuda_eliteSize = cuda.to_device(np.array([eliteSize], dtype=np.int32))
    cuda_crossoverRate = cuda.to_device(np.array([crossoverRate], dtype=np.float64))
    cuda_mutationRate = cuda.to_device(np.array([mutationRate], dtype=np.float64))
    cuda_N_target = cuda.to_device(np.array([N_target], dtype=np.int32))
    cuda_sat_samplings = cuda.to_device(sat_samplings)
    cuda_sat_attitudes = cuda.to_device(sat_attitudes)
    cuda_sat_vectors = cuda.to_device(sat_vectors)
    cuda_sat_occultated = cuda.to_device(sat_occultated)
    cuda_tar_prioritys = cuda.to_device(tar_prioritys)
    cuda_tar_vectors = cuda.to_device(tar_vectors)
    cuda_is_adaptive = cuda.to_device(np.array([METADATA['adaptive']]))

    start = time.perf_counter()

    threads_per_block = 128
    blocks_per_grid = (popSize + (threads_per_block - 1)) // threads_per_block

    # states
    np.random.seed(METADATA['seed'])
    state_seeds = np.random.rand(3)
    states = []
    for i in range(len(state_seeds)):
        states.append(create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=state_seeds[i]))


    init_population[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_N_target, states[0])

    for i in range(num_generations + 1):

        eval_genomes_kernel[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_fitnesses, cuda_popSize,
                                                                cuda_N_target, cuda_sat_samplings, cuda_sat_attitudes,
                                                                cuda_sat_vectors, cuda_sat_occultated,
                                                                cuda_tar_prioritys, cuda_tar_vectors)

        if i < num_generations:

            sort_chromosomes[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_fitnesses,
                                                                 cuda_sorted_chromosomes,
                                                                 cuda_sorted_fitnesses, cuda_fitnessTotal)
            if METADATA['adaptive']:
                adaptive_mechanism_kernel[blocks_per_grid, threads_per_block](cuda_crossoverRate, cuda_mutationRate)

            # Crossover And Mutation
            crossover[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_sorted_chromosomes,
                                                          cuda_sorted_fitnesses, cuda_rouletteWheel,
                                                          states[1], cuda_popSize, cuda_eliteSize, cuda_fitnessTotal,
                                                          cuda_crossoverRate, cuda_is_adaptive)

            # update the tmp_fitnesses
            if METADATA['adaptive']:
                eval_genomes_kernel[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_tmp_fitnesses,
                                                                        cuda_popSize,
                                                                        cuda_N_target, cuda_sat_samplings,
                                                                        cuda_sat_attitudes,
                                                                        cuda_sat_vectors, cuda_sat_occultated,
                                                                        cuda_tar_prioritys, cuda_tar_vectors)

            mutation[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_tmp_fitnesses, cuda_N_target,
                                                         states[2], cuda_mutationRate, cuda_is_adaptive)

    end = time.perf_counter()
    show_fitness_pairs = []
    chromosomes = cuda_chromosomes.copy_to_host()
    fitnesses = cuda_fitnesses.copy_to_host()

    for i in range(len(chromosomes)):
        show_fitness_pairs.append([chromosomes[i], fitnesses[i]])
    fitnesses = list(reversed(sorted(fitnesses)))  # fitnesses now in descending order
    show_sorted_pairs = list(reversed(sorted(show_fitness_pairs, key=lambda x: x[1])))
    best_allocation = show_sorted_pairs[0][0]
    best_fitness = show_sorted_pairs[0][1]
    return best_allocation, best_fitness, end - start
