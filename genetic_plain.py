import numpy as np
import random
import math
from target_track_gym.metadata import METADATA
import time



def eval_genomes_plain(chromosomes, fitnesses, popSize, N_satellite, N_target, sat_samplings, sat_attitudes,
                       sat_vectors, sat_occultated, tar_prioritys, tar_vectors):
    for i in range(popSize):
        target_fitness_list = [[0 for k in range(N_satellite)] for j in range(N_target)]
        # chromosomes.shape [popsize, chrom_size]
        chrom = chromosomes[i]

        for tar_id in range(N_target):
            positions = [index for index, every_tar in enumerate(chrom) if every_tar == tar_id]

            if len(positions) > 0:
                for sat_id in positions:
                    occultated = sat_occultated[sat_id][tar_id]

                    if occultated:
                        fitness = 0
                    else:
                        # cal motion factor
                        s_x, s_y = sat_vectors[sat_id]
                        t_x, t_y = tar_vectors[tar_id]
                        dot_product = s_x * t_x + s_y * t_y
                        norm_product = math.sqrt(s_x ** 2 + s_y ** 2) * math.sqrt(
                            t_x ** 2 + t_y ** 2)
                        angle = math.acos(dot_product / norm_product) / math.pi * 180
                        motion_factor = 1 - angle / 180
                        fitness = sat_samplings[sat_id] * sat_attitudes[sat_id][tar_id] * motion_factor * tar_prioritys[
                            tar_id]

                    target_fitness_list[tar_id][sat_id] = fitness

            target_fitness_list[tar_id] = max(target_fitness_list[tar_id])
        fitnesses[i] = sum(target_fitness_list)


def adaptive_mechanism_plain(crossoverRate, mutationRate):
    crossoverRate *= 0.995
    mutationRate *= 0.995
    return crossoverRate, mutationRate


def next_generation_plain(chromosomes, fitnesses, tmp_fitnesses, N_target, sat_samplings, sat_attitudes, sat_vectors,
                          sat_occultated, tar_prioritys, tar_vectors, eliteSize, crossoverRate, mutationRate):
    is_adaptive = METADATA['adaptive']


    fitness_pairs = []
    fitnessTotal = 0.0
    for i in range(len(chromosomes)):
        fitness_pairs.append([chromosomes[i], fitnesses[i][0]])
        fitnessTotal += fitnesses[i][0]

    sorted_pairs = np.array(list(reversed(sorted(fitness_pairs, key=lambda x: x[1]))), dtype=object)
    sorted_chromsomos, sorted_fitnesses = sorted_pairs[:, 0:1], sorted_pairs[:, 1:2]
    sorted_chromsomos = np.array([np.array(chrom[0], dtype=np.int64) for chrom in sorted_chromsomos]).reshape(
        chromosomes.shape)

    new_chromosomes = np.zeros(shape=chromosomes.shape, dtype=np.int32)

    # create roulette wheel from relative fitnesses for fitness-proportional selection
    rouletteWheel = []
    fitnessProportions = []
    for i in range(len(chromosomes)):
        fitnessProportions.append(float(sorted_fitnesses[i] / fitnessTotal))
        if i == 0:
            rouletteWheel.append(fitnessProportions[i])
        else:
            rouletteWheel.append(rouletteWheel[i - 1] + fitnessProportions[i])




    for i in range(0, eliteSize):
        # Add new chromosome to next population
        new_chromosomes[i] = sorted_chromsomos[i]


    # Crossover (Generate new population with children of selected chromosomes)
    for i in range(eliteSize, len(chromosomes), 2):

        # Fitness Proportional Selection
        spin1 = random.uniform(0, 1)  # A random float from 0.0 to 1.0
        spin2 = random.uniform(0, 1)  # A random float from 0.0 to 1.0

        j = 0
        while (rouletteWheel[j] <= spin1):
            j += 1

        k = 0
        while (rouletteWheel[k] <= spin2):
            k += 1

        genome_copy1 = sorted_chromsomos[j]  # Genome of parent 1
        genome_copy2 = sorted_chromsomos[k]  # Genome of parent 2

        # create child genome from parents (crossover)
        index1 = random.randint(0, len(genome_copy1) - 1)
        index2 = random.randint(0, len(genome_copy2) - 1)

        child_sequence1 = []
        child_sequence2 = []

        fitness_c = np.max([sorted_fitnesses[j], sorted_fitnesses[k]])
        fitness_max = np.max(sorted_fitnesses)
        fitness_avg = np.mean(sorted_fitnesses)

        if is_adaptive:
            crossoverRate_ = crossoverRate * (fitness_max - fitness_c) / (fitness_max - fitness_avg)
            crossoverRate_ = np.clip(crossoverRate_, 0.1, 1)
        else:
            crossoverRate_ = crossoverRate

        if random.uniform(0, 1) < crossoverRate_:
            for y in range(len(genome_copy1)):
                if y <= math.floor(len(genome_copy1) / 2):
                    child_sequence1.append(genome_copy1[(index1 + y) % len(genome_copy1)])
                    child_sequence2.append(genome_copy2[(index2 + y) % len(genome_copy2)])
                else:
                    child_sequence2.append(genome_copy1[(index1 + y) % len(genome_copy1)])
                    child_sequence1.append(genome_copy2[(index2 + y) % len(genome_copy2)])

            new_chromosomes[i], new_chromosomes[i + 1] = child_sequence1, child_sequence2

        else:
            new_chromosomes[i], new_chromosomes[i + 1] = genome_copy1, genome_copy2



    if is_adaptive:
        # update the tmp_fitnesses
        eval_genomes_plain(new_chromosomes, tmp_fitnesses, chromosomes.shape[0], chromosomes.shape[1], N_target,
                           sat_samplings,
                           sat_attitudes, sat_vectors, sat_occultated, tar_prioritys, tar_vectors)

    fitness_max = np.max(tmp_fitnesses)
    fitness_avg = np.mean(tmp_fitnesses)

    # mutation
    for i in range(len(chromosomes)):
        fitness_m = tmp_fitnesses[i][0]
        if is_adaptive:
            mutationRate_ = mutationRate * (fitness_max - fitness_m) / (fitness_max - fitness_avg)
            mutationRate_ = np.clip(mutationRate_, 0.05, 1)

        else:
            mutationRate_ = mutationRate
        for a in range(len(chromosomes[0])):
            if random.uniform(0, 1) < mutationRate_:
                random_gene = random.randint(0, N_target - 1)
                new_chromosomes[i][a] = random_gene

    # Replace old chromosomes with new
    for i in range(len(chromosomes)):
        for j in range(len(chromosomes[0])):
            chromosomes[i][j] = new_chromosomes[i][j]
