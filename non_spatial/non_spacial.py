import numpy as np
import random as rand
import matplotlib.pyplot as plt

# genes
L = 14
N = 3000
P = 0


# reproduction
Delta = 1

#mutation
mu0 = N/10
muinf = 10**-6
generations = 4e4
tau = generations/5
delta = generations/10


# fitness
Alpha = 2
R = 2
Beta = 2
Gamma = 6 
J =1



#initialize n lists of bits (genomes) in an array
# following a binomial distribution
def init_individual():
    arr = np.zeros(L)
    arr[:np.random.binomial(L,P)]  = 1
    np.random.shuffle(arr)
    return arr

def init_pop():
	pop[:, 1:] = init_individual() #np.random.randint(2, size=(N, L))
	phenotypes(pop)

def phenotypes(popu):
	popu[:, 0] = np.sum(popu[:, 1:], axis=1)


def reproduction():
	index = np.random.randint(N,size=(N))
	partners = []
	for i, phenotype in enumerate(pop[index, 0]):
		poss_partners = np.argwhere((pop[:, 0] <= phenotype + Delta) & (pop[:, 0] >= phenotype - Delta) ).flatten()
		partners.append(rand.choice(poss_partners))

	which_partner = np.random.randint(2, size=(N,L))
	offspring[:, 2:] = np.where(which_partner, pop[:, 1:], pop[partners, 1:])
	mutate()
	phenotypes(offspring[:, 1:])

def switch(value):
	return 0 if value else 1

def mutate():
	genes = np.random.randint(2, L + 2, size=N)
	chances = np.random.rand(N)
	for i, gene, chance in zip(np.arange(N), genes, chances):
		if chance < Mu: 
			offspring[i, gene] = switch(offspring[i, gene]) 
	

def H_0(x):
	return np.exp( (- 1/Beta) * (x/Gamma) ** Beta)

def fitness():
	unique, nx_t = np.unique(offspring[:, 1], return_counts=True)
	px_t = nx_t/N
	H0 = H_0(unique)
	H = [H0[i] - J * sum(p*np.exp( (- 1/Alpha) * np.abs((x - y)/R) **Alpha ) 
				for y,p in zip(unique, px_t) if y < x+R and y > x-R) 
			for i, x in enumerate(unique)]
	
	A = np.exp(H)
	A /=np.mean(A)
	A_dict = {key:value for key, value in zip(unique, A)}
	offspring[:, 0] = list(map(lambda x: A_dict[x], offspring[:, 1]))
 
def new_gen():
	survivors = []
	np.random.shuffle(offspring)
	while len(survivors) < N:
		chances = np.random.rand(N)
		survived = np.argwhere(chances < offspring[:, 0]).flatten()
		survivors.extend(survived)
	pop[:,:] = offspring[survivors[:N], 1:]


def Mu_t(t):
	return (mu0 - muinf)*(1 - np.tanh((t-tau)/delta))/2 + muinf



pop = np.zeros((N, L + 1))
offspring = np.zeros((N, L + 2))

Mu_all = Mu_t(np.arange(generations))
init_pop()

for i, Mu in enumerate(Mu_all):
	reproduction()
	mutate()
	fitness()
	new_gen()
	uniq, counts = np.unique(offspring[:, 1], return_counts=True)
	if not i % 1000:	
		print(uniq)
		print(counts)

# np.random.seed(0)

# a = np.random.randint(20, size=(10,4))	
# b = np.random.randint(10, size=10)

# c = np.zeros((5,3))
# mask = a[:, 0] < b
# d = a[mask]
# print(d)

# c[3:3+len(d)] = d[:, :3]
# print(c)
