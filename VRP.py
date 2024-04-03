import sys
import math
import random
import numpy as np
from numpy import sin, cos, arccos, pi


class GA:
    def __init__(self, total_p, city, dis_list, size, gen_rate, mut_rate,car,car_pivot):
        # Currently adopted methods include: Tournament Selection / Two-Point Crossover / Swap Mutation
        self.total_p = total_p # Total population count
        self.city = city # Number of cities
        self.dis_list = dis_list # Distance between each pair of points
        self.size = size # Population size
        self.gen_rate = gen_rate # Crossover rate
        self.mut_rate = mut_rate # Mutation rate
        self.car = car # Number of cars
        self.car_pivot = car_pivot # Standard distance each car should travel      
        self.population = self.init_population() # Generate initial population
        self.answer = [[0]*(self.city+self.car-2),sys.maxsize] # Store the best solution
        
    def init_population(self): # Initialize population
        population = []
        for index in range(self.size):
            chromosome = list(range(2,self.city+1))
            random.shuffle(chromosome) # Randomly shuffle elements in the list (one of the functions provided by the random module)
            chromosome = self.init_cut(chromosome) # Add cut points for each segment of the route, forming a gene
            # Each individual consists of: serial number within the population, gene, fitness (score), whether acted as parent
            population.append([index,chromosome,self.Fitness_distance(self.cut_road_list(chromosome)),False])
        return population
    
    def init_cut(self,gene): # Divide the route into n segments according to the number of cars
        # random_num: a random number between 0 and 1, representing the weight of running all routes for each car
        # cut_list: number of cities each car should visit
        # last: length of the last segment (subtract 1 because the first city (starting point) is not counted)
        random_num,cut_list,last = [],[0 for i in range(self.car)],self.city-1
        
        for i in range(last) :
            random_car = random.randint(0, len(cut_list)-1)
            cut_list[random_car] += 1

        cut_point = self.city # Cut points are represented by numbers greater than the city index
        for i in range(len(cut_list)):
            index = 0 # Insert cut point at the index position
            for a in range(i):
                index += cut_list[a] # Calculate the index based on the number of cities each car should visit
            index += i
            gene.insert(index,cut_point) # Insert cut point at the index position
            cut_point += 1 # Increment the cut point representation number
        del gene[0]
        return gene
 
    def cut_road_list(self,gene): # Divide each route into an array
        result,temp = [],[] 
        for x in gene:
            if x > self.city: # If the current number is a cut point, add the current array to the result list and create a new array to record the next segment
                result.append(temp)
                temp = []
            else:
                temp.append(x)
        if temp: # Add to the result only when temp is not empty
            result.append(temp)
        return result

    def Fitness_distance(self,gene_list): # Calculate score (total distance)
        score = 0
        over_limit_car = 0
        each_route_score = []
        for gene in gene_list: # Multiple genes in the same chromosome, consisting of multiple lists of routes (one car per route)
            road_score = 0 # Total distance of the route
            if (gene != []): # When the car needs to depart
                for i in range(len(gene)-1):
                    # Calculate the Euclidean distance between gene[i] and gene[i+1], and add it to road_score
                    a,b = sorted([gene[i],gene[i+1]])
                    road_score += self.dis_list[a-1][b-a-1]
                # Calculate the distance from the starting point to the first city, and from the last city to the starting point, and add them to road_score
                a,b,c = 1,gene[0],gene[-1]
                road_score += self.dis_list[a-1][b-a-1]
                road_score += self.dis_list[a-1][c-a-1]
                each_route_score.append(road_score)
                
            # limit: limit condition (each car cannot travel more than a certain distance) / weight: penalty weight, indicating the proportion of exceeding the limit to be penalized
            limit = self.car_pivot # Used as a reference for other parameters
            weight, min_limit, weight_over_limit_car = limit*80, limit/1.75, limit/6
            if road_score > limit: # Penalty for excessive distance traveled by a single route
                over_limit_car += weight_over_limit_car # The more cars exceed the limit, the less penalty they receive (because it may be due to a poorly set limit that causes everyone to exceed)
                score += (road_score + ((road_score-limit)*weight / over_limit_car))
            elif road_score < min_limit : # Penalty for too little distance traveled by a single route
                score += (road_score + ((limit-road_score)*weight))
            else : # No penalty for too little or too much distance traveled
                score += road_score
            # Penalize routes with too many or too few cities
            city_limit = round((self.city - 1) / self.car) # On average, each car should visit how many cities
            w_city = limit * 10
            if len(gene) > city_limit * 1.25 : # Too many cities visited (exceeding average)
                score += (len(gene) - city_limit) * w_city
            elif len(gene) < city_limit/2 : # Too few cities visited (less than half of average)
                score += (len(gene) - city_limit) * w_city * 2 # Penalty for too few visits is higher than for too many
        
        # Penalize excessive standard deviation of distances
        each_route_score = np.array(each_route_score) 
        std = np.std(each_route_score, ddof=1) # Calculate standard deviation
        std_limit = 3 # Standard deviation limit, the higher it is, the greater the difference in distance between cars
        if std > std_limit :
            score += (std-std_limit) * weight * 500 # Excessive standard deviation is heavily penalized, because having a balanced distance is important
        return score
    
    def Selection_Tournament(self): # Use tournament selection method (may select duplicate chromosomes)
        new_population = []
        for index in range(self.size):
            player = random.choices(self.population, k=8) # Randomly select a few from the population based on the previously created probability list, here k should be based on city
            fit_list = [gene[2] for gene in player] # Fitness of selected chromosomes
            winner = player[fit_list.index(min(fit_list))] # Select the one with the minimum fitness
            new_population.append([index,winner[1],winner[2],False])    
        return new_population
    
    def Choose_parent(self): # Select genes of two chromosomes as parents
        remaining_population = [gene for gene in self.population if gene[3] == False] # Select those who haven't been chosen as parents yet
        choose = random.choices(remaining_population, k=2) # Randomly select two from the population
        self.population[choose[0][0]][3],self.population[choose[1][0]][3] = True,True  # Mark those who have been chosen as parents
        return choose
    
    def Crossover(self, parent1, parent2): # Crossover method, using Two-Point Crossover (PMX)
        gene_len = self.city+self.car-2
        child = [0]*gene_len
        start, end = sorted(random.sample(range(gene_len), 2)) # Randomly select two points for cutting
        for i in range(start, end+1): # Let the first part equal to parent 1
            child[i] = parent1[1][i]
        j = end + 1
        for i in range(gene_len):
            if parent2[1][i] not in child: # If this number is not in child yet
                if j == gene_len: # If already reached the end, start from the beginning
                    j = 0
                child[j] = parent2[1][i] # Add to child
                j += 1 # Move to the next position to add to child
        return child

    def Mutation(self, chromosome): # Mutation method, using Swap Mutation
        i, j = random.sample(range(self.city), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome
        
    def Evolve(self): # Evolution process
        for _ in range(math.ceil(self.total_p/self.size)): # Total number of generations      
            new_population = []
            self.population = self.Selection_Tournament() # Selection
            for i in range(0,self.size,2): # Crossover
                parent1, parent2 = self.Choose_parent() # Randomly select two chromosomes that have not been chosen as parents from the population
                if random.random() < self.gen_rate: # If crossover occurs
                    child1,child2 = self.Crossover(parent1, parent2),self.Crossover(parent2, parent1) # Generate new offspring chromosomes
                    # Add new offspring chromosomes to the new population
                    new_population.append([i,child1,self.Fitness_distance(self.cut_road_list(child1)),False])
                    new_population.append([i+1,child2,self.Fitness_distance(self.cut_road_list(child2)),False])
                else: # If no crossover, fill in the original genes
                    new_population.append([i,parent1[1],parent1[2],False])
                    new_population.append([i+1,parent2[1],parent2[2],False])
            
            for i in range(self.size): # Mutation
                if random.random() < self.mut_rate:
                    gene = self.Mutation(new_population[i][1])
                    new_population[i] = [i,gene,self.Fitness_distance(self.cut_road_list(gene)),False]
                   
            self.population = new_population # Update population
            f_list = [gene[2] for gene in self.population] # Find the best solution with the minimum distance
            best_fit = min(f_list)
            if ( best_fit < self.answer[1]): # If it's smaller, replace it
                self.answer = [self.population[f_list.index(best_fit)][1],best_fit]
        return self.answer
      
def main():
    # time: Number of repetitions of GA / total_p: Total population count / size: Population size
    # gen_rate: Crossover rate / mut_rate: Mutation rate / file_name: File name containing coordinates / car: Maximum number of cars
    #time,total_p,size,gen_rate,mut_rate,file_name,car = 10,10000,100,0.92,0.12,'Berlin52.txt',5  
    time,total_p,size,gen_rate,mut_rate,file_name,car = 30,10000,100,0.92,0.12,'delivery.txt',15
    with open(file_name, 'r') as f: 
        city = int(f.readline().strip()) # Get the number of cities from the first line of the file
        # Use list comprehension to convert each line's coordinates to a tuple, and store all tuples in the list city_coordinate
        city_coordinate = [tuple(map(float, line.strip().split()[1:])) for line in f.readlines()] 

    print("\n======= Current Test Parameter Data =======")
    print("file_name : ",file_name)
    print("Time     =",time,"   city     =",city,
          "\ntotal_p  =",total_p,"gen_rate =",gen_rate,
          "\nsize     =",size,"  mut_rate =",mut_rate)
    print("================================")

    ans_list,fit_list,dis_list = [],[],init_dis_list(city,city_coordinate) # Arrays to store results
    car_pivot = each_car_pivot(dis_list[0], car) # Standard distance each car should travel on average
    for i in range(time):
        print("Exam :",i+1,"time")
        test = GA(total_p,city,dis_list,size,gen_rate,mut_rate,car,car_pivot) # Population count, number of cities, distances between cities, population size, crossover rate, mutation rate, number of cars
        ans = test.Evolve() # Return the best chromosome of this generation
        ans_list.append(test.cut_road_list(ans[0])) # Divide each route into an array
        fit_list.append(ans[1]) # Fitness of this chromosome
        print("GBEST FIT =",ans[1])
        print("================================")
   
    mean_value = sum(fit_list) / len(fit_list)
    std_value = math.sqrt(sum((x - mean_value) ** 2 for x in fit_list) / (len(fit_list) - 1))
    print("GBEST Fitness      = ", round(min(fit_list),4))
    print("Average            = ", round(mean_value,4))
    print("Standart Deviation = ", round(std_value,6))
    
    best_answer,i,total_dis = ans_list[fit_list.index(min(fit_list))],1,0
    print("\n================== GBEST tour detailed ==================")
    for road_list in best_answer:
        if road_list != []:
            dis = count_distance(dis_list,road_list)
            print("Car "+str(i)+" Route    :",*road_list)
            print("      Distance :",dis)
            print("      Route counts :", len(road_list))
            print("==========================================================")
            i += 1
            total_dis += dis
    print("Total Distance =",total_dis) 

def each_car_pivot(city_dis, car) :
    avg_each_city_dis = sum(city_dis) / len(city_dis)
    avg_car_city = len(city_dis) / car
    return avg_each_city_dis * avg_car_city

def init_dis_list(city,city_coordinate): # Calculate the distance between each point (can directly call the distance between two points when calculating the score to save redundant calculation time)
    list = []
    for i in range(city-1): # Current city
        dis = []
        for j in range(i+1,city): # Next city 
            dis.append(euclidean_distance(city_coordinate[i], city_coordinate[j]))
        list.append(dis)
    return list
    
def euclidean_distance(a, b): # Calculate the Euclidean distance between a and b (straight-line distance between two points)
    #return math.sqrt(sum([(a[i] - b[i])**2 for i in range(len(a))])) # Calculate the straight-line distance between two points
    return getDistanceBetweenPointsNew(a[0],a[1],b[0],b[1]) # Calculate the straight-line distance between two points using latitude and longitude

def rad2deg(radians):
    degrees = radians * 180 / pi
    return degrees

def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians

def getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2):
    theta = longitude1 - longitude2
    distance = 60 * 1.1515 * rad2deg(
        arccos(
            (sin(deg2rad(latitude1)) * sin(deg2rad(latitude2))) + 
            (cos(deg2rad(latitude1)) * cos(deg2rad(latitude2)) * cos(deg2rad(theta)))
        )
    )
    return round(distance * 1.609344, 2)

def count_distance(dis_list,gene): # Calculate distance
        score = 0
        for i in range(len(gene)-1):
            # Calculate the Euclidean distance between gene[i] and gene[i+1], and add it to score
            a,b = sorted([gene[i],gene[i+1]])
            score += dis_list[a-1][b-a-1]
        # Calculate the Euclidean distance between the starting point and the first city, and between the last city and the starting point, and add them to score
        a,b,c = 1,gene[0],gene[-1]
        score += dis_list[a-1][b-a-1]
        score += dis_list[a-1][c-a-1]
        return score

if __name__ == '__main__':
    main()
