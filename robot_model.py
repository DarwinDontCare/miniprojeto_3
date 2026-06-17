import numpy as np

class Robot6DOF:
    def __init__(self):
        # Limites articulares em radianos
        self.joint_bounds = np.array([
            [-2.96, 2.96],  # J1
            [-1.74, 2.35],  # J2
            [-2.09, 2.87],  # J3
            [-3.22, 3.22],  # J4
            [-2.09, 2.09],  # J5
            [-6.28, 6.28]   # J6
        ])

        # Parâmetros DH (alpha, a, d)
        self.dh_params = [
            (np.pi/2,  0.070,  0.352), # Elo 1
            (0.0,      0.360,  0.0),   # Elo 2
            (np.pi/2,  0.090,  0.0),   # Elo 3
            (-np.pi/2, 0.0,    0.380), # Elo 4
            (np.pi/2,  0.0,    0.0),   # Elo 5
            (0.0,      0.0,    0.065)  # Elo 6
        ]
        
    def get_all_joint_transforms(self, theta):
        """ Retorna uma lista com todas as matrizes T acumuladas (T0_1, T0_2...T0_6) """
        T = np.eye(4)
        transforms = [np.copy(T)] # Base frame (origem 0,0,0)
        
        for i in range(6):
            alpha, a, d = self.dh_params[i]
            th = theta[i]
            ct, st = np.cos(th), np.sin(th)
            ca, sa = np.cos(alpha), np.sin(alpha)
            
            A = np.array([
                [ct, -st*ca,  st*sa, a*ct],
                [st,  ct*ca, -ct*sa, a*st],
                [0,   sa,     ca,     d],
                [0,   0,      0,      1]
            ])
            T = T @ A
            transforms.append(np.copy(T))
            
        return transforms

    def forward_kinematics(self, theta):
        """ Retorna apenas a pose do efetuador final (T0_6) formatada """
        all_T = self.get_all_joint_transforms(theta)
        T = all_T[-1]
        return np.array([T[0,3], T[1,3], T[2,3], T[0,2], T[1,2], T[2,2]])

    def elastic_repair(self, theta):
        """ Reparo por Memória Elástica """
        repaired = np.copy(theta)
        for i in range(6):
            low, high = self.joint_bounds[i]
            if repaired[i] < low or repaired[i] > high:
                mid = (low + high) / 2.0
                clipped = np.clip(repaired[i], low, high)
                repaired[i] = 0.8 * clipped + 0.2 * mid
        return repaired