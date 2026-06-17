import numpy as np
import time
import webbrowser
import threading
try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False

class RobotVisualizer:
    def __init__(self, robot_model):
        self.available = MESHCAT_AVAILABLE
        if not self.available:
            print("AVISO: Biblioteca 'meshcat' não encontrada. O 3D será pulado.")
            return
            
        self.robot = robot_model
        self.vis = meshcat.Visualizer()
        print(f"\nVisualizador Meshcat iniciado em: {self.vis.url()}")
        
        try:
            webbrowser.open(self.vis.url()) 
        except:
            pass
        
        self.setup_scene()

    def create_axes_triad(self, node, scale=0.1):
        """ Cria eixos XYZ tridimensionais (R, G, B) usando cilindros finos """
        thick = 0.003
        node["y"].set_object(g.Cylinder(scale, radius=thick), g.MeshBasicMaterial(color=0x00FF00))
        node["y"].set_transform(tf.translation_matrix([0, scale/2, 0]))
        
        node["x"].set_object(g.Cylinder(scale, radius=thick), g.MeshBasicMaterial(color=0xFF0000))
        Rx = tf.rotation_matrix(-np.pi/2, [0, 0, 1])
        node["x"].set_transform(tf.translation_matrix([scale/2, 0, 0]) @ Rx)
        
        node["z"].set_object(g.Cylinder(scale, radius=thick), g.MeshBasicMaterial(color=0x0000FF))
        Rz = tf.rotation_matrix(np.pi/2, [1, 0, 0])
        node["z"].set_transform(tf.translation_matrix([0, 0, scale/2]) @ Rz)

    def setup_scene(self):
        self.vis["/Grid"].set_property("visible", True)
        
        joint_geom = g.Sphere(0.03)
        self.joint_mat = g.MeshLambertMaterial(color=0xAAAAAA, reflectivity=0.5)
        link_geom = g.Cylinder(0.1, radius=0.01) 
        self.link_mat = g.MeshLambertMaterial(color=0x3366FF)

        for i in range(7):
            self.vis[f"robot/joint_{i}"].set_object(joint_geom, self.joint_mat)
            if i > 0:
                self.vis[f"robot/link_{i}"].set_object(link_geom, self.link_mat)

        t_geom = g.Sphere(0.02)
        t_mat = g.MeshBasicMaterial(color=0xFF0000)
        self.vis["target/point"].set_object(t_geom, t_mat)
        
        self.create_axes_triad(self.vis["target/axes"], scale=0.15)
        self.create_axes_triad(self.vis["robot/joint_6/axes"], scale=0.1)

    def set_target_pose(self, target_pose_vec):
        if not self.available: return
        pos = target_pose_vec[0:3]
        self.vis["target"].set_transform(tf.translation_matrix(pos))

    def update_robot_pose(self, theta):
        if not self.available: return
        theta_rep = self.robot.elastic_repair(theta)
        transforms = self.robot.get_all_joint_transforms(theta_rep)
        
        for i in range(7):
            T_i = transforms[i]
            self.vis[f"robot/joint_{i}"].set_transform(T_i)
            
            if i > 0:
                T_prev = transforms[i-1]
                p_prev = T_prev[0:3, 3]
                p_curr = T_i[0:3, 3]
                v = p_curr - p_prev
                length = np.linalg.norm(v)
                
                if length < 0.001:
                    self.vis[f"robot/link_{i}"].set_property("visible", False)
                    continue
                
                self.vis[f"robot/link_{i}"].set_property("visible", True)
                midpoint = (p_prev + p_curr) / 2.0
                y_axis = np.array([0, 1, 0])
                v_norm = v / length
                
                if np.allclose(y_axis, v_norm):
                    R = np.eye(3)
                elif np.allclose(y_axis, -v_norm):
                    R = tf.rotation_matrix(np.pi, [1, 0, 0])[0:3, 0:3]
                else:
                    axis = np.cross(y_axis, v_norm)
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.dot(y_axis, v_norm))
                    R = tf.rotation_matrix(angle, axis)[0:3, 0:3]
                    
                T_link = tf.translation_matrix(midpoint)
                T_link[0:3, 0:3] = R
                S = np.diag([1.0, length / 0.1, 1.0, 1.0])
                
                self.vis[f"robot/link_{i}"].set_transform(T_link @ S)

    def animate_solution(self, final_theta, duration=3.0):
        """ Anima a solução em loop contínuo (Ping-Pong) com controle de Play/Pause """
        if not self.available: return
        
        self.anim_running = True
        self.anim_playing = True

        def keyboard_listener():
            print("\n" + "="*60)
            print("CONTROLES DA ANIMAÇÃO 3D:")
            print(" -> Pressione ENTER (vazio) para Pausar / Continuar")
            print(" -> Digite 'q' + ENTER para Fechar o programa")
            print("="*60 + "\n")
            
            while self.anim_running:
                command = input().strip().lower()
                if command == 'q':
                    self.anim_running = False
                    break
                else:
                    self.anim_playing = not self.anim_playing
                    status = "REPRODUZINDO" if self.anim_playing else "PAUSADO"
                    print(f"[{status}] (Pressione ENTER para alternar ou 'q' para sair)")

        listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
        listener_thread.start()
        
        print(f"Iniciando loop de animação...")
        frames = 60
        start_theta = np.zeros(6)
        
        t = 0.0
        direction = 1
        step = 1.0 / frames
        
        try:
            while self.anim_running:
                if self.anim_playing:
                    interp_theta = (1 - t) * start_theta + t * final_theta
                    self.update_robot_pose(interp_theta)
                    
                    t += direction * step
                    
                    if t >= 1.0:
                        t = 1.0
                        direction = -1
                    elif t <= 0.0:
                        t = 0.0
                        direction = 1
                
                time.sleep(duration / frames)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.anim_running = False
            print("\nLoop de animação finalizado.")