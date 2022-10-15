import numpy as np
import pandas as pd
from prepare_dataset import *
from trade_operation import *

from concurrent.futures import ProcessPoolExecutor


class StrategyOptimizer:

    def __init__(self,
            fitness_function,
            n_generations,
            generation_size,
            n_genes,
            gene_ranges,
            mutation_prob,
            gene_mutation_prob,
            n_select_best,
            df,
            sl_target,
            stake_amt,
            initial_wallet,
            commission,
            good_gene= 0
            ):
        
        '''
        fitness_function,       = to build
        n_generations           = 100
        generation_size         = 300
        n_genes                 = pending
        gene_ranges             = pending, shud be in range i.e. (3,500), (3,50), (0,100)... for each gene
        mutation_prob           = 0.3
        gene_mutation_prob      = 0.5
        n_select_best           = 20
        '''

        self.fitness_function = fitness_function
        self.n_generations = n_generations
        self.generation_size = generation_size
        self.n_genes = n_genes
        self.gene_ranges =gene_ranges
        self.mutation_prob= mutation_prob
        self.gene_mutation_prob=gene_mutation_prob
        self.n_select_best= n_select_best
        self.df = df
        #self.tp_target = tp_target
        self.sl_target =sl_target
        self.stake_amt =stake_amt
        self.initial_wallet =initial_wallet
        self.good_gene = good_gene
        self.commission = commission

    
    def create_individual(self):
        """Create individual"""
        individual = []

        for i in range(self.n_genes):
            gene = np.random.randint(self.gene_ranges[i][0],self.gene_ranges[i][1]+1) #for each gene, random between max and min range
            individual.append(gene)
        return individual


    def create_population(self, population_size):
        """Create population"""
        population = []

        if self.good_gene:
            population_size_ = population_size - 1
            population.append(self.good_gene)
        else:
            population_size_ = population_size

        for i in range(population_size_):
            population.append(self.create_individual())
        return population



    def mate_parents(self, parents, n_offspring): #input = parents pool, select 2 parents randomly, mate for n_offsprings count
        """Mate parents and returns n_offspring individuals"""
        n_parents = len(parents) 
        offsprings = []

        for i in range(n_offspring):
            random_parent1 = parents[np.random.randint(0, n_parents)]
            random_parent2 = parents[np.random.randint(0, n_parents)]
        
            #do random mask to exchange genes for children
            parent1_mask = np.random.randint(0,2,size=np.array(random_parent1).shape)
            parent2_mask = np.logical_not(parent1_mask) #inverse of 1st parent

            #born child
            child = np.add(np.multiply(random_parent1, parent1_mask), np.multiply(random_parent2, parent2_mask))
            offsprings.append(child)
        return offsprings
        


    def mutate_individual(self, individual):  #individual is just range of strategies params
        """mutate individual"""
        new_individual = []

        #mutate for every gene
        for i in range(self.n_genes):
            gene = individual[i]
            if np.random.random() < self.gene_mutation_prob:
                #two ways of mutation, brute force and range finder
                if np.random.random() < 0.5:
                    gene = np.random.randint(self.gene_ranges[i][0], self.gene_ranges[i][1]+1)
                
                else:
                    left_range = self.gene_ranges[i][0]
                    right_range = self.gene_ranges[i][1]
                    gene_dist = right_range - left_range
                    if gene_dist == 1:
                        gene = np.random.randint(0,2)

                    else: 

                        x = individual[i] + gene_dist / 3 * (2*np.random.random()-1) #move randomly between +/- 33% from where the gene is
                        
                        if x > right_range:
                            x = (x - right_range) + left_range #if exceed max limit, add from left range
                        elif x < left_range:
                            x = right_range - (left_range - x)  #if exceed min limit, minus from right range

                        gene = int(x)
                
            #no hit prob
            new_individual.append(gene)
        return new_individual



    def mutate_population(self, population):
        """mutate population"""
        mutated_pop = []

        for individual in population:
            new_individual = individual
            # every individual got chance to be selected for mutation candidate
            # every selected invidiual got another chance to actually mutate
            if np.random.random() < self.mutation_prob:
                new_individual = self.mutate_individual(individual)
            
            mutated_pop.append(new_individual)
        return mutated_pop


    def fitness_fn_loop(self, input):
        """for multiprocess"""
        idx, individual = input
        # gene_rng, gene_mask, tp = individual[:14], individual[14:-1], individual[-1]/100
        # long_sig, short_sig = populate_entry_trend(self.df, gene_rng, gene_mask)
        # trade_data = buy_sell(self.df, long_sig, short_sig, tp, self.sl_target, self.stake_amt)
        # _, netprofit, _, _ = self.fitness_function(trade_data, self.initial_wallet)
        netprofit = backtesting(self.df, individual, self.sl_target, wallet=self.initial_wallet, commission=self.commission, trade_on_close=True)

        return (idx, netprofit, individual)


    def select_best(self, population, n_best):
        """select the best n_best individuals out of a population"""
        fitness = []

        with ProcessPoolExecutor(max_workers=10) as executor:
        #for idx, individual in enumerate(population):
            for r in executor.map(self.fitness_fn_loop, enumerate(population)):
                
            # gene_rng, gene_mask = individual[:7], individual[7:]
            # buy_sig = populate_entry_trend(df, gene_rng, gene_mask)
            # trade_data = buy_sell(df, buy_sig, tp_target, sl_target, stake_amt)
            # trade_cnt, netprofit, profit_perc, winrate = self.fitness_function(trade_data, initial_wallet)
            #idx, netprofit, individual = self.fitness_fn_loop(idx, individual, df, initial_wallet, tp_target, sl_target, stake_amt)
                fitness.append(r)
                print(r)
        
        #select top n_best, put top best candidates into parent pools for possible mating.
        cost_tmp = pd.DataFrame(fitness).sort_values(by=1, ascending=False).reset_index(drop=True)
        selected_parents_idx = list(cost_tmp.iloc[:n_best, 0])
        selected_parents = [parent for idx,parent in enumerate(population) if idx in selected_parents_idx] 

        print('Best is {}, average is {}, worst is {}'.format(cost_tmp[1].max(), cost_tmp[1].mean(), cost_tmp[1].min()))
        print('Best genes is ', cost_tmp.iloc[0][2])
        print(cost_tmp.iloc[0])


        return cost_tmp, selected_parents


    def run_genetic_algo(self):
        """"""

        # first create entire population
        parent_gen = self.create_population(self.generation_size)

        # for each generation
        for i in range(self.n_generations):
            print('>> Generation - ',i)

            #select n_select_best as parents
            start_gen = time.time()
            _, top_parent_pool = self.select_best(parent_gen, self.n_select_best)
            print("Time taken for select best:", time.time() - start_gen)

            #pick 2 and mate them to create new generation
            start = time.time()
            new_gen = self.mate_parents(top_parent_pool, self.generation_size)
            print("Time taken for mutate:", time.time() - start)

            #mutate the newly create population
            start = time.time()
            parent_gen = self.mutate_population(new_gen)
            print("Time taken for mutate pop:", time.time() - start)
            print("Total time for 1 gen:", time.time() - start_gen)


        # supposedly best children (strategies) after x generations
        cost_table, best_children = self.select_best(parent_gen, 10)
        return cost_table, best_children



if __name__ == '__main__':

    
    initial_wallet = 20000
    stake_amt = 7000
    sl_target= 0.015
    commission = 0.006

    if not os.path.isfile('data/all_feat_working.csv'):
        df = retrieve_data()
        df = generate_features(df)
        save_file(df)
        df = df.set_index(pd.to_datetime(df['date'].apply(lambda x: unix2date(x/1000))))
        print('Job done. File saved.')

    else:
        df = pd.read_csv('data/all_feat_working.csv')
        df = df.set_index(pd.to_datetime(df['date'].apply(lambda x: unix2date(x/1000))))


    gene_rng = [(3,100), (3,200), (1,100), (1,100), (1,100), (1,80),  (1,80),  (3,200),  #buy params
                (3,200), (3,100), (1,100), (1,100), (1,100), (30,90), (30,90), (3,200),  #sell params
                (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1),
                (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (0,1), (1,6)]
    SO = StrategyOptimizer(
        fitness_function = fitness_fn, 
        n_generations = 10, #50, 
        generation_size = 300, #400, 
        n_genes= 37, 
        gene_ranges = gene_rng, 
        mutation_prob = 0.4, 
        gene_mutation_prob = 0.6, 
        n_select_best=8,
        df = df,
        sl_target = sl_target,
        stake_amt = stake_amt,
        initial_wallet=initial_wallet,
        commission = commission,
        good_gene = 0 #[ 41,  54,  18,  33,  79,  18,  42, 126, 108,   9,  94,  27,  20,
        # 76,  78,  70,   0,   0,   1,   0,   0,   1,   0,   1,   0,   0,
        #  0,   0,   1,   1,   1,   1,   1,   0,   0,   1,  11]
        )


    cost_table, best_children = SO.run_genetic_algo()
    cost_table.to_csv('data/gen_result_sep.csv')
    print(best_children)
