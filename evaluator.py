import numpy as np

class ObjectiveEvaluator:
    def __init__(self, robot, target_pose):
        self.robot = robot
        self.target = target_pose
        self.fevals = 0
        self.history = []
        
        # Guardas de estado para coletar o melhor resultado real
        self.best_error = float('inf')
        self.best_theta = None

    def evaluate(self, theta):
        self.fevals += 1
        theta_repaired = self.robot.elastic_repair(theta)
        current_pose = self.robot.forward_kinematics(theta_repaired)
        
        # Erro quadrático médio
        error = np.mean((current_pose - self.target) ** 2)
        self.history.append((self.fevals, error))
        
        # Atualiza a melhor solução vista nesta instância
        if error < self.best_error:
            self.best_error = error
            self.best_theta = np.copy(theta_repaired)
            
        return error