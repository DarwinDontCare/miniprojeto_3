import numpy as np
import matplotlib.pyplot as plt
import time

from robot_model import Robot6DOF
from visualizer import RobotVisualizer
from evaluator import ObjectiveEvaluator
from algorithms import run_genetic_algorithm, run_pso, run_bas, run_memetic_ga_bas

if __name__ == "__main__":
    robot = Robot6DOF()
    
    # Alvo espacial (X, Y, Z, nx, ny, nz)
    target_pose = np.array([0.4, 0.3, 0.2, 0.0, 0.0, -1.0])
    
    # Inicializa o Visualizador 3D Meshcat
    visualizer = RobotVisualizer(robot)
    visualizer.set_target_pose(target_pose)
    visualizer.update_robot_pose(np.zeros(6)) # Pose Home inicial
    
    algorithms = {
        "GA": run_genetic_algorithm,
        "PSO": run_pso,
        "BAS": run_bas,
        "Híbrido (GA-BAS)": run_memetic_ga_bas
    }
    
    results = {}
    num_runs = 20 
    max_fevals = 2500
    
    # Variáveis para capturar a melhor solução global absoluta do benchmark inteiro
    absolute_best_fitness = float('inf')
    absolute_best_theta = None
    best_algo_name = ""

    print(f"\nIniciando Simulações Comparativas Justas ({num_runs} Runs, budget={max_fevals} FEs)...")
    print("O Matplotlib abrirá primeiro. O navegador 3D abrirá em seguida.")

    plt.figure(figsize=(10, 6))
    
    for name, algo_func in algorithms.items():
        run_fitnesses = []
        all_histories = []
        
        for run in range(num_runs):
            np.random.seed(run * 42) # Reprodutibilidade
            evaluator = ObjectiveEvaluator(robot, target_pose)
            
            # Executa o algoritmo atual
            algo_func(evaluator, robot, max_fevals=max_fevals)
            
            final_fit = evaluator.best_error
            run_fitnesses.append(final_fit)
            all_histories.append(evaluator.history)
            
            # Coleta o vencedor absoluto para a animação
            if final_fit < absolute_best_fitness:
                absolute_best_fitness = final_fit
                absolute_best_theta = evaluator.best_theta
                best_algo_name = name

        results[name] = {
            "mean": np.mean(run_fitnesses),
            "std": np.std(run_fitnesses)
        }
        
        # Processamento das curvas de convergência para o gráfico
        fevals_axis = np.arange(1, max_fevals + 1)
        interpolated_curves = []
        for hist in all_histories:
            fevals_h = [h[0] for h in hist]
            fit_h = [h[1] for h in hist]
            best_so_far = np.minimum.accumulate(fit_h)
            interp = np.interp(fevals_axis, fevals_h, best_so_far)
            interpolated_curves.append(interp)
            
        mean_curve = np.mean(interpolated_curves, axis=0)
        plt.plot(fevals_axis, mean_curve, label=f"{name} (Média)")

    # Exibição do Gráfico de Convergência
    plt.yscale("log")
    plt.title("Comparação Justa Fixando Avaliações de Função Objetivo", fontsize=12)
    plt.xlabel("Avaliações da Função Objetivo (Recurso Computacional)", fontsize=10)
    plt.ylabel("Erro da Cinemática Inversa (Log)", fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    print("\nExibindo gráfico de convergência Matplotlib...")
    plt.show(block=False)
    plt.pause(1)
    plt.savefig("convergence_comparison.png", dpi=300)
    print("Gráfico salvo como 'convergence_comparison.png'.")

    # Execução da Animação 3D com a melhor configuração real obtida
    print("\n" + "="*60)
    print("FASE DE VISUALIZAÇÃO 3D")
    print("="*60)
    print(f"Melhor algoritmo global: {best_algo_name}")
    print(f"Menor erro alcançado: {absolute_best_fitness:.2e}")

    if visualizer.available and absolute_best_theta is not None:
        input("\nPressione ENTER para iniciar a animação 3D no navegador...")
        visualizer.animate_solution(absolute_best_theta, duration=4.0)
        print(f"\nVerifique o navegador em: {visualizer.vis.url()}")
        print("Mantenha o terminal aberto. Ctrl+C para encerrar o servidor Meshcat.")
        
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            print("\nEncerrando servidor visualizador.")
    else:
        print("\nVisualização 3D indisponível ou nenhuma solução encontrada.")