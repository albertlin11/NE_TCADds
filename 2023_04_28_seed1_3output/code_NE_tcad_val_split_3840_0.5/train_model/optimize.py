import pandas as pd
import numpy as np
import tensorflow as tf
import random
import copy
import os
import time
import glob
from NE_model import create_architecture
from NE_model import NE_model
from mlp_model import run_mlp
from util import run_trend_gen
from util import gen_loss_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

seed = 1
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def create_genome():
    process_architecture = create_architecture(5, 4, 6)

    return dict([('process_architecture', process_architecture)])

def crossover(genome1, genome2):
    newg1 = {}
    newg2 = {}
    # process_architecture
    hidden_a1 = copy.deepcopy(genome1['process_architecture'][0])
    hidden_a2 = copy.deepcopy(genome2['process_architecture'][0])
    output_a1 = copy.deepcopy(genome1['process_architecture'][1])
    output_a2 = copy.deepcopy(genome2['process_architecture'][1])
    k1 = list(hidden_a1.keys())
    k1.sort()
    k2= list(hidden_a2.keys())
    k2.sort()
    layer_a1 = k1[-1][0]
    layer_a2 = k2[-1][0]
    crosspoint_a1 = random.randint(1, layer_a1-1)
    crosspoint_a2 = random.randint(1, layer_a2-1)
    newhidden1_1 = {}
    newhidden1_2 = {}
    newhidden2_1 = {}
    newhidden2_2 = {}

    for key in k1:
        if key[0] <= crosspoint_a1:
            newhidden1_1[key] = hidden_a1[key]
        else:
            newhidden1_2[key] = hidden_a1[key]
    for key in k2:
        if key[0] <= crosspoint_a2:
            newhidden2_1[key] = hidden_a2[key]
        else:
            newhidden2_2[key] = hidden_a2[key]

    newhidden1_1.update(newhidden2_2)
    newhidden2_1.update(newhidden1_2)
    newg1['process_architecture'] = [newhidden1_1, output_a2]
    newg2['process_architecture'] = [newhidden2_1, output_a1]

    return newg1, newg2

def create_model(genome):
   # Construct: Model
    model, loss_info, val_loss_info = NE_model(genome)

    return model, loss_info, val_loss_info

def get_record():
    """It will return mape"""
    mape_raw = []
    params_value = []
    for filename in glob.glob('process_record.txt'):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            for line in f.readlines():
                if "mape_raw " in line:
                    mape = line.replace("\n", "").replace("mape_raw = ", "")
                    mape_raw.append((filename.replace(".txt", ""),float(mape)))
                if "trainable_params " in line:
                    params = line.replace("\n", "").replace("trainable_params = ", "")
                    params_value.append(("trainable_params",float(params)))
            f.close()

    return [mape_raw, params_value]

def fitness(record):
    #convert record to tuple for comparison
    mape = sum([i[1] for i in record[0]])

    return mape

def performance():
    """It will return mape"""
    mape_raw = []
    for filename in glob.glob('process_record.txt'):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            for line in f.readlines():
                if "test_mape_performance " in line:
                    mape = line.replace("\n", "").replace("test_mape_performance = ", "")
                    mape_raw.append((filename.replace(".txt", ""),float(mape)))
            f.close()

    return mape_raw

class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.models, self.loss_info, self.val_loss_info = create_model(self.genome)
        self.record = get_record()
        self.fitness = fitness(self.record)
        self.performance = performance()

def init_population(pop_size):
    Population_0 = []
    for i in range(pop_size):
        genome_ini = create_genome()
        Population_0.append(Individual(genome_ini))

    return Population_0

#tournament selection ->return better genome(only 1)
def selection(pop):
    s = random.sample(pop, 2)
    # python compare tuples lexicographically
    if s[0].fitness < s[1].fitness:
        best = s[0]
    else:
        best = s[1]
    return best

#N:population size, input population, return mate_pool for reproduction
def sel_parents(pop, N):
    mate_pool = []
    for i in range(N//2):
        parent1 = selection(pop)
        parent2 = selection(pop)
        mate_pool.append([parent1, parent2])
    return mate_pool

def offspring(mate_pool):
    child_population = []
    for parents in mate_pool:
        g1 = parents[0].genome
        g2 = parents[1].genome
        childg1, childg2 = crossover(g1, g2)
        child1 = Individual(childg1)
        child2 = Individual(childg2)
        child_population.append(child1)
        child_population.append(child2)
    return child_population

#Environment selection. Inviduals with better fitness score are allowed to survive, others will be
#eliminated so that the population will remain the same in each generation.
def env_select(pop,pop_size):
    pop.sort(key = lambda x: x.fitness)
    pop=pop[0:pop_size]
    return pop

#Evaluate the whole population
def evaluation(pop):
    evol_record = []
    evol_results = []
    evol_performance = []
    evol_loss_info = []
    evol_val_loss_info = []
    for x in pop:
        evol_record.append(x.record)
        evol_results.append(x.fitness)
        evol_performance.append(x.performance)
        evol_loss_info.append(x.loss_info)
        evol_val_loss_info.append(x.val_loss_info)
    return [evol_record, evol_results, evol_performance, evol_loss_info, evol_val_loss_info]

#Run the evolution and return the fittest individual
def run_evolution(gen_limit,pop_size):
    generation = {}
    records = []
    results = []
    pop = init_population(pop_size)
    k = 0
    generation[k] = pop
    eva = evaluation(pop)
    records.append(eva[0])
    results.append(eva[1])
    df_records = pd.DataFrame(eva[0])
    df_results = pd.DataFrame(eva[1])
    df_performance = pd.DataFrame(eva[2])
    loss_info_gen = eva[3]
    val_loss_info_gen = eva[4]
    df_records.to_csv("./evolution_log/records/records" + str(k) + "th_gen.csv")
    df_results.to_csv("./evolution_log/fitness/fitness" + str(k) + "th_gen.csv")
    df_performance.to_csv("./evolution_log/performance/performance" + str(k) + "th_gen.csv")
    loss_info_gen = np.array(loss_info_gen)
    val_loss_info_gen = np.array(val_loss_info_gen)
    np.save("./evolution_log/records/loss_info" + str(k) + "th_gen.npy", loss_info_gen)
    np.save("./evolution_log/records/val_loss_info" + str(k) + "th_gen.npy", val_loss_info_gen)
    if not os.path.exists("./evolution_log/tf_model/tf_" + str(k) + "th_gen"):
        os.makedirs("./evolution_log/tf_model/tf_" + str(k) + "th_gen")
    for l in range(len(generation[k])):
        mod = generation[k][l].models
        mod.save("./evolution_log/tf_model/tf_" + str(k) + "th_gen/model" + str(l) + ".h5")
    for i in range(gen_limit):
        k += 1
        mate_pool = sel_parents(pop,pop_size)
        child_pop = offspring(mate_pool)
        pop = child_pop+pop
        pop = env_select(pop,pop_size)
        generation[k] = pop
        eva = evaluation(pop)
        records.append(eva[0])
        results.append(eva[1])
        df_records = pd.DataFrame(eva[0])
        df_results = pd.DataFrame(eva[1])
        df_performance = pd.DataFrame(eva[2])
        loss_info_gen = eva[3]
        val_loss_info_gen = eva[4]
        df_records.to_csv("./evolution_log/records/records" + str(k) + "th_gen.csv")
        df_results.to_csv("./evolution_log/fitness/fitness" + str(k) + "th_gen.csv")
        df_performance.to_csv("./evolution_log/performance/performance" + str(k) + "th_gen.csv")
        np.save("./evolution_log/records/loss_info" + str(k) + "th_gen.npy", loss_info_gen)
        np.save("./evolution_log/records/val_loss_info" + str(k) + "th_gen.npy", val_loss_info_gen)
        if not os.path.exists("./evolution_log/tf_model/tf_" + str(k) + "th_gen"):
            os.makedirs("./evolution_log/tf_model/tf_" + str(k) + "th_gen")
        for l in range(len(generation[k])):
            mod = generation[k][l].models
            mod.save("./evolution_log/tf_model/tf_" + str(k) + "th_gen/model" + str(l) + ".h5")
        best_genome = [str(pop[0].genome)]
        df_best_genome = pd.DataFrame(best_genome)
        df_best_genome.to_csv("./evolution_log/genome/genome" + str(k) + "th_gen.csv")
        best_models = pop[0].models
        best_models.save("./evolution_log/tf_model/model" + str(k) + "th_gen.h5")

    best_individual = pop[0]
    return best_individual, records, results, generation



#%%
start = time.process_time()
best, gen_records, gen_results, gen_pop = run_evolution(10, 10)
best_models = best.models
best_genome = [str(best.genome)]
fitnessscore = best.fitness
stop = time.process_time()
processtime = stop-start
# run mlp model
run_mlp()
# generate trend data and loss data csv
run_trend_gen(3840)
gen_loss_data(10)

df_best_genome = pd.DataFrame(best_genome)
df_best_genome.to_csv("./evolution_log/best/best_genome.csv")
best_models.save("./evolution_log/best/tf_model/best_model.h5")
df_evol_records = pd.DataFrame(gen_records)
df_evol_results= pd.DataFrame(gen_results)
df_evol_records.to_csv("./evolution_log/best/records/evolution_records.csv")
df_evol_results.to_csv("./evolution_log//best/fitness/evolution_fitness.csv")
