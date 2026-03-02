import sys
import subprocess
from pathlib import Path

from scripts.dataset_info import mostrar_info_dataset
from scripts.visualizacoes.vis_supervisionado import dashboard_supervisionado
from scripts.visualizacoes.vis_nao_supervisionado import dashboard_nao_supervisionado
from scripts.visualizacoes.vis_reforco import dashboard_reforco

SCRIPT_DIR = Path(__file__).resolve().parent / "machine_learning"


def esperar_voltar(prompt: str = "\nPressione Enter para voltar ao menu..."):
    """Pausa simples para voltar ao menu principal."""
    try:
        input(prompt)
    except KeyboardInterrupt:
        print()


def menu_principal():
    print("\n===== PROJETO FINAL — TÓPICOS EM CIÊNCIA DE DADOS =====")
    print("1 - Informações do dataset")
    print("2 - Visualizações e Resultados (Machine Learning)")
    print("3 - Executar modelo supervisionado")
    print("4 - Executar clusterização (não supervisionado)")
    print("5 - Executar aprendizagem por reforço")
    print("0 - Sair")


def menu_visualizacoes():
    print("\n=== VISUALIZAÇÕES E RESULTADOS ===")
    print("1 - Aprendizagem Supervisionada")
    print("2 - Aprendizagem Não Supervisionada")
    print("3 - Aprendizagem por Reforço")
    print("0 - Voltar")


def executar_script(script_name: str):
    """Executa um script dentro de scripts/machine_learning usando o mesmo Python do venv."""
    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        print(f"Arquivo não encontrado: {script_path}")
        return

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        print("Modelo executado com sucesso.\n")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {script_path}: {e}")


def main():
    while True:
        menu_principal()
        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            mostrar_info_dataset()
            esperar_voltar()

        elif opcao == "2":
            while True:
                menu_visualizacoes()
                sub = input("Escolha uma opção: ").strip()

                if sub == "1":
                    dashboard_supervisionado()
                    esperar_voltar()

                elif sub == "2":
                    dashboard_nao_supervisionado()
                    esperar_voltar()

                elif sub == "3":
                    dashboard_reforco()
                    esperar_voltar()

                elif sub == "0":
                    break

                else:
                    print("Opção inválida.")

        elif opcao == "3":
            executar_script("ml_supervisionado.py")
            esperar_voltar()

        elif opcao == "4":
            executar_script("ml_clusterizacao.py")
            esperar_voltar()

        elif opcao == "5":
            executar_script("ml_reforco.py")
            esperar_voltar()

        elif opcao == "0":
            print("Encerrando o programa.")
            break

        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()