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

    # Dicionários para guardar as curvas médias de cada tipo de gráfico
    mean_curves_fevals = {}
    mean_curves_iters = {}
    
    for name, algo_func in algorithms.items():
        run_fitnesses = []
        all_histories = []
        run_times = []
        all_iter_curves = []
        
        for run in range(num_runs):
            np.random.seed(run * 42) # Reprodutibilidade
            evaluator = ObjectiveEvaluator(robot, target_pose)
            
            # --- MEDIÇÃO DE TEMPO INÍCIO ---
            start_time = time.time()
            algo_func(evaluator, robot, max_fevals=max_fevals)
            elapsed_time = time.time() - start_time
            run_times.append(elapsed_time)
            # -------------------------------
            
            final_fit = evaluator.best_error
            run_fitnesses.append(final_fit)
            all_histories.append(evaluator.history)
            
            # Coleta o vencedor absoluto para a animação
            if final_fit < absolute_best_fitness:
                absolute_best_fitness = final_fit
                absolute_best_theta = evaluator.best_theta
                best_algo_name = name

        # CORREÇÃO 1: Coleta das 5 métricas estatísticas exigidas no Requisito 4
        results[name] = {
            "Melhor (Min)": np.min(run_fitnesses),
            "Pior (Max)": np.max(run_fitnesses),
            "Média": np.mean(run_fitnesses),
            "Desvio Padrão": np.std(run_fitnesses),
            "Tempo Médio (s)": np.mean(run_times)
        }
        
        # --- PROCESSAMENTO GRÁFICO 1: CONVERGÊNCIA POR FEs ---
        fevals_axis = np.arange(1, max_fevals + 1)
        interpolated_fevals_curves = []
        for hist in all_histories:
            fevals_h = [h[0] for h in hist]
            fit_h = [h[1] for h in hist]
            best_so_far = np.minimum.accumulate(fit_h)
            interp = np.interp(fevals_axis, fevals_h, best_so_far)
            interpolated_fevals_curves.append(interp)
            
        mean_curves_fevals[name] = np.mean(interpolated_fevals_curves, axis=0)

        # --- PROCESSAMENTO GRÁFICO 2: CONVERGÊNCIA POR ITERAÇÕES ---
        # Mapeamento do tamanho da iteração de cada algoritmo para o eixo X
        # GA (Steady State) = 1 FE por iteração; PSO = 30 FEs; BAS = 3 FEs; Híbrido ~= 1 FE
        fevals_per_iter = 30 if name == "PSO" else (3 if name == "BAS" else 1)
        max_iters = int(max_fevals / fevals_per_iter)
        
        interpolated_iter_curves = []
        iters_axis = np.arange(1, max_iters + 1)
        for curve in interpolated_fevals_curves:
            # Seleciona os pontos correspondentes ao fim de cada iteração real
            indices = (iters_axis * fevals_per_iter) - 1
            indices = np.clip(indices, 0, len(curve) - 1)
            interpolated_iter_curves.append(curve[indices])
            
        mean_curves_iters[name] = (iters_axis, np.mean(interpolated_iter_curves, axis=0))

    # --- IMPRESSÃO DA TABELA COMPARATIVA (Requisito 7) ---
    print("\n" + "="*50)
    print("TABELA COMPARATIVA DOS RESULTADOS (20 RUNS)")
    print("="*50)
    for name, metrics in results.items():
        print(f"\nAlgoritmo: {name}")
        for metric_name, val in metrics.items():
            print(f"  {metric_name}: {val:.4e}" if "Tempo" not in metric_name else f"  {metric_name}: {val:.4f}s")

    plt.figure(1, figsize=(10, 5))
    for name, curve in mean_curves_fevals.items():
        plt.plot(np.arange(1, max_fevals + 1), curve, label=f"{name}")
    plt.yscale("log")
    plt.title("Conversão por Avaliações da Função Objetivo (Comparação Justa)", fontsize=11)
    plt.xlabel("Avaliações da Função Objetivo (FEs)", fontsize=10)
    plt.ylabel("Erro da Cinemática Inversa (Log)", fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig("convergence_by_fevals.png", dpi=300)

    plt.figure(2, figsize=(10, 5))
    for name, (axis, curve) in mean_curves_iters.items():
        plt.plot(axis, curve, label=f"{name}")
    plt.yscale("log")
    plt.title("Convergência por Iterações/Gerações (Escalas Diferentes)", fontsize=11)
    plt.xlabel("Número de Iterações / Gerações", fontsize=10)
    plt.ylabel("Erro da Cinemática Inversa (Log)", fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig("convergence_by_iterations.png", dpi=300)

    print("\nGráficos salvos com sucesso ('convergence_by_fevals.png' e 'convergence_by_iterations.png').")
    plt.show(block=False)

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