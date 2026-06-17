import numpy as np

def run_genetic_algorithm(evaluator, robot, max_fevals=3000, pop_size=40):
    bounds = robot.joint_bounds
    pop = np.random.uniform(bounds[:,0], bounds[:,1], (pop_size, 6))
    fitness = np.array([evaluator.evaluate(ind) for ind in pop])
    while evaluator.fevals < max_fevals:
        idx1, idx2 = np.random.randint(0, pop_size, 2)
        parent1 = pop[idx1] if fitness[idx1] < fitness[idx2] else pop[idx2]
        idx3, idx4 = np.random.randint(0, pop_size, 2)
        parent2 = pop[idx3] if fitness[idx3] < fitness[idx4] else pop[idx4]
        alpha = np.random.rand()
        child = alpha * parent1 + (1 - alpha) * parent2
        if np.random.rand() < 0.3:
            child += np.random.normal(0, 0.1, 6)
        child = robot.elastic_repair(child)
        child_fit = evaluator.evaluate(child)
        worst_idx = np.argmax(fitness)
        if child_fit < fitness[worst_idx]:
            pop[worst_idx] = child
            fitness[worst_idx] = child_fit

def run_pso(evaluator, robot, max_fevals=3000, num_particles=30):
    bounds = robot.joint_bounds
    X = np.random.uniform(bounds[:,0], bounds[:,1], (num_particles, 6))
    V = np.zeros_like(X)
    pbest = np.copy(X)
    pbest_fit = np.array([evaluator.evaluate(ind) for ind in X])
    gbest = pbest[np.argmin(pbest_fit)]
    gbest_fit = np.min(pbest_fit)
    w, c1, c2 = 0.5, 1.5, 1.5
    while evaluator.fevals < max_fevals:
        for i in range(num_particles):
            if evaluator.fevals >= max_fevals: break
            r1, r2 = np.random.rand(6), np.random.rand(6)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] = robot.elastic_repair(X[i] + V[i])
            fit = evaluator.evaluate(X[i])
            if fit < pbest_fit[i]:
                pbest[i] = X[i]
                pbest_fit[i] = fit
                if fit < gbest_fit: 
                    gbest = X[i]
                    gbest_fit = fit

def run_bas(evaluator, robot, max_fevals=3000):
    bounds = robot.joint_bounds
    x = np.random.uniform(bounds[:,0], bounds[:,1])
    d0, step, eta = 0.5, 0.5, 0.95
    best_x = np.copy(x)
    best_fit = evaluator.evaluate(x)
    while evaluator.fevals < max_fevals:
        dir_vec = np.random.randn(6)
        dir_vec /= np.linalg.norm(dir_vec)
        x_left = robot.elastic_repair(x + d0 * dir_vec)
        x_right = robot.elastic_repair(x - d0 * dir_vec)
        f_left = evaluator.evaluate(x_left)
        f_right = evaluator.evaluate(x_right)
        x = robot.elastic_repair(x - step * dir_vec * np.sign(f_left - f_right))
        fit = evaluator.evaluate(x)
        if fit < best_fit: 
            best_fit = fit
            best_x = np.copy(x)
        step *= eta
        d0 *= eta + 0.01

def run_memetic_ga_bas(evaluator, robot, max_fevals=3000, pop_size=40):
    bounds = robot.joint_bounds
    pop = np.random.uniform(bounds[:,0], bounds[:,1], (pop_size, 6))
    fitness = np.array([evaluator.evaluate(ind) for ind in pop])
    while evaluator.fevals < max_fevals:
        idx1, idx2 = np.random.randint(0, pop_size, 2)
        parent1 = pop[idx1] if fitness[idx1] < fitness[idx2] else pop[idx2]
        idx3, idx4 = np.random.randint(0, pop_size, 2)
        parent2 = pop[idx3] if fitness[idx3] < fitness[idx4] else pop[idx4]
        child = robot.elastic_repair(0.5 * parent1 + 0.5 * parent2)
        if np.random.rand() < 0.3: 
            child += np.random.normal(0, 0.1, 6)
        child = robot.elastic_repair(child)
        child_fit = evaluator.evaluate(child)
        if evaluator.fevals % 200 == 0:
            best_idx = np.argmin(fitness)
            refined_x = np.copy(pop[best_idx])
            dir_vec = np.random.randn(6)
            dir_vec /= np.linalg.norm(dir_vec)
            xl = robot.elastic_repair(refined_x + 0.05 * dir_vec)
            xr = robot.elastic_repair(refined_x - 0.05 * dir_vec)
            fl, fr = evaluator.evaluate(xl), evaluator.evaluate(xr)
            refined_x = robot.elastic_repair(refined_x - 0.02 * dir_vec * np.sign(fl - fr))
            refined_fit = evaluator.evaluate(refined_x)
            if refined_fit < fitness[best_idx]: 
                pop[best_idx] = refined_x
                fitness[best_idx] = refined_fit
        worst_idx = np.argmax(fitness)
        if child_fit < fitness[worst_idx]: 
            pop[worst_idx] = child
            fitness[worst_idx] = child_fit