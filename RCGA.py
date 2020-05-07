import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math

'''
    @Class Real-Coded Genetic Algorithm for minimizing an objective function
    @return - Minima of the function
'''   
class RCGA:
    def __init__(self, n_pop=100, max_gen=100, selection_method="TS", tournament_size=2, p_c=0.9, p_m=0.01, crossOver_method="2x", mutation_method ="MPT", initial_guess=0, UB=1, LB=-1):
        self.n_pop = n_pop
        self.gen = 0
        self.max_gen = max_gen
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.num_select = n_pop
        self.n_digits = initial_guess
        self.crossOver_method = crossOver_method
        self.mutation_method = mutation_method
        self.fitness_gen = []
        self.gen = 1
        self.p_c = p_c
        self.p_m = p_m
        self.fit_elite = 0
        self.elite = 0
        self.error_elite = 0
        self.fitness_function = 0
        self.gbest_param = 0
        self.param_iter = []
        self.min_value = 0
        self.UB = UB
        self.LB = LB

    '''
    This method is where the search is carried out by calling the RCGA methods
    @param - Function to be minimized
    @return - Minima of the function
    '''                                                                           
    def search(self, fun):

        A_fit = [0]*self.n_pop
        self.n_digits = fun.getDimension()
        A_pop = RCGA.InitializePop(self)
        
        while self.gen <= self.max_gen:
                        
            for i in range(0, self.n_pop):
                yhat = RCGA.predict(self, A_pop[:,i], fun)
                A_fit[i] = RCGA.evaluateFitness(self, yhat)

            A_fit_s = RCGA.scaleFitness(self, A_fit)  
            selected = RCGA.selection(self, A_fit_s)
            A_pop = RCGA.setPop(self, A_pop, selected)
            A_pop = RCGA.crossOver(self, A_pop, A_fit)
            RCGA.elitism(self, A_fit, A_pop)
            
            self.fitness_gen.append(1/max(A_fit))
            self.param_iter.append(self.gbest_param)
            print("Function value:", 1/max(A_fit) )
            RCGA.setMutationProbability(self)

            
            self.gen += 1

        self.min_value =  1 /max(A_fit) 

    '''
    The population is initialized in this method
    @return - numpy array of size N x n_pop
    ''' 
    def InitializePop(self):
        
        A = zeros(self.n_digits, self.n_pop)
        a = self.LB
        b = self.UB

        for i in range(0,self.n_digits):
            for j in range(0,self.n_pop):
                A[i][j] = (b - a) * rand.random() + a          
            
        return np.array(A)

    '''
    Generic interface for selection
    The selection is implemented either as RouletteWheel or TournamentSelection
    @return - List of indices of the selected chromosomes in the population
    ''' 
    def selection(self, A_fit):

        if self.selection_method == "TS":
            return RCGA.tournamentSelection(self, A_fit)
        elif self.selection_method == "RW":
            return RCGA.rouletteWheelSelection(self, A_fit)
    '''
    This method implements the Tournament Selection
    @return - List of indices of the selected chromosomes in the population
    ''' 
    def tournamentSelection(self, A_fit):
        
        selected = []
        
        while len(selected) < self.n_pop:

            p = np.random.permutation(self.n_pop)

            for i in range(0, self.n_pop - self.tournament_size, self.tournament_size):
                
                group = A_fit[p[i:(i+self.tournament_size)]]
                best = np.argmax(group)     
                selected.append(p[best + i])
                
        return selected[0:self.n_pop]

    '''
    This method implements the RouletteWheelSelection (Placeholder)
    @return - List of indices of the selected chromosomes in the population
    ''' 
    def rouletteWheelSelection(self, A_fit):
        print("Nothing here yet.")
        return 0

    '''
    This method sets the selected population
    @return - Selected individuals in the population
    ''' 
    def setPop(self, A_pop, selected):
        return A_pop[:,selected]

    '''
    Evaluates the fitness based on the function value
    @return - Fitness of an individual
    ''' 
    def evaluateFitness(self, yhat):
        return 1 / yhat

    '''
    Interface for CrossOver. Implements the Arithmetic CrossOver
    @return - Reproduced population
    ''' 
    def crossOver(self, A_pop, A_fit):
        
        a = -0.25
        b = 1.25

        for i in range(0, self.n_pop - 1):

            parent_1 = A_pop[:,i]
            parent_1_temp = deepCopy(parent_1)
            parent_2 = A_pop[:,i+1]
            
            if self.crossOver_method == "2x":
                r1 = rand.randint(0, self.n_digits)
                r2 = rand.randint(0, self.n_digits)
            elif self.crossOver_method == "1x":
                r1 = rand.randint(0, self.n_digits)
                r2 = self.n_digits

            alfa = (b - a)*rand.random() + a
            if rand.random() < self.p_c:
                
                for j in range(0, self.n_digits):
                    parent_1[j] = alfa*parent_2[j] + (1 - alfa)*parent_1[j]
                    parent_2[j] = alfa*parent_1_temp[j] + (1 - alfa)*parent_2[j]
            
            if rand.random() < self.p_m:
                
                parent_1 = RCGA.mutation(self, parent_1, A_pop)

            if rand.random() < self.p_m:
                
                parent_2 = RCGA.mutation(self, parent_2, A_pop)
            
            A_pop[:,i] = parent_1
            A_pop[:,i+1] = parent_2

        return A_pop
    
    '''
    Interface for mutation. Implements either the DE mutation scheme or Mäkinen-Periaux-Toivonen mutation.
    @return - Mutated individual
    ''' 
    def mutation(self, parent_1, A_pop):

        if self.mutation_method == "DE":
            return RCGA.DE_mutation(self,parent_1, A_pop)
        elif self.mutation_method == "MPT":
            return RCGA.MPT_mutation(self, parent_1, A_pop)
    
    '''
    Implements the Mäkinen-Periaux-Toivonen mutation.
    @return - Mutated individual
    '''
    def MPT_mutation(self, parent_1, A_pop):
            
            l = self.LB
            u = self.UB
            p = 2

            t_m = []
            t_mat = (parent_1 - l) / (u - l)
            r = rand.random()

            for t in t_mat:
                if  r < t:
                    t_m.append(t - t * ((t-r) / t)**p)
                elif r == t:
                    t_m.append(t)
                elif r > t:
                    t_m.append(t + (1- t) * ((t-r) / (1 - t))**p)

            t_m = np.array(t_m)
            return ( 1 - t_m ) * l + t_m * u

    '''
    Implements the DE mutation scheme.
    @return - Mutated individual
    '''
    def DE_mutation(self, parent_1, A_pop):

        x_s = A_pop[:, rand.randint(0, self.n_pop - 1)]
        x_t = A_pop[:, rand.randint(0, self.n_pop - 1)]
        x_l = A_pop[:, rand.randint(0, self.n_pop - 1)]

        mu = 0.5
        sigma = 0.5
        F = np.random.normal(mu, sigma)

        return x_s + F * (x_t - x_l) + F*(self.gbest_param - x_s)

    '''
    Set mutation probability according to the generation and number of digits
    '''
    def setMutationProbability(self):
        self.p_m = 1/(2 + (self.n_digits - 2) / (self.max_gen - 1)*self.gen)

    '''
    Select the most fit chrosome and update the attribute gbest_param.
    '''
    def elitism(self, A_fit, A_pop):
        
        if max(A_fit) > self.fit_elite:
            A_fit_argmax = np.argmax(A_fit)
            self.gbest_param = A_pop[:,A_fit_argmax] 
            self.fit_elite = max(A_fit)

    '''
    Scale fitness to classical probability
    @return Scaled fitness of the population
    '''
    def scaleFitness(self, A_fit):
        A_fit = A_fit / sum(A_fit)
        return A_fit

    '''
    Evaluate the objective function value
    @param X - Vector of parameters, f = function object
    @return Function value with parameters X
    '''
    def predict(self, X, f):
        
        return f.eval(X)

'''
Create a vector of zeros (n x k)
@return list of zeros
'''
def zeros(n, k=1):
    if k > 1:   
        return [[0 for i in range(k)] for j in range(n)]
    else:
        return [0]*n

'''
Make a deep copy of list
@return Deep copy of x
'''
def deepCopy(x):
    y = np.copy(x)
    return y


'''
Example of how to use the solver
Currently uses the Rastrigin function for testing purposes
'''
class testFunction:

    def __init__(self, name, N=20):
        self.N = N
        self.name = name

    def eval(self, X):

        if self.name == "Rastrigin":
            
            f_sum = 0
            for i in range(0, self.N):
                f_sum = f_sum + ( X[i]**2 - 10*math.cos(2*3.1459*X[i]) )
            return 10*self.N + f_sum

    def getDimension(self):
        return self.N


def main():
    ### Initialize the solver###
    ### RCGA = Real-Coded Genetic Algorithm
    solver = RCGA(n_pop=200, max_gen = 500, p_c=0.9, p_m=0.1, tournament_size=2, crossOver_method="2x", mutation_method="MPT", UB=5.12, LB=-5.12) 

    ###Create a test function object###
    f = testFunction("Rastrigin", N=40) 
    
    ###Search the minima###
    solver.search(f)

    ###Display results###
    print("\n\nX = ", solver.gbest_param)
    print("\n\nf(X):", solver.min_value, "\n\n")
    
    ###Plot results####
    p = plt.plot(solver.fitness_gen)
    plt.ylabel('f(x)')
    plt.xlabel('iteration')
    plt.title('convergence graph')
    plt.show()

if __name__ == "__main__":
    main()