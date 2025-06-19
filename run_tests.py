#!/usr/bin/env python
"""
Script para executar os testes do projeto
"""
import subprocess
import sys
import argparse


def run_all_tests():
    """Executa todos os testes"""
    print("🧪 Executando todos os testes...")
    return subprocess.call([sys.executable, "-m", "pytest", "-v"])


def run_performance_tests():
    """Executa apenas os testes de desempenho"""
    print("📊 Executando testes de desempenho do modelo...")
    return subprocess.call([
        sys.executable, "-m", "pytest", 
        "tests/test_model_performance.py", 
        "-v", 
        "-s"  # Mostra print statements
    ])


def run_unit_tests():
    """Executa apenas os testes unitários"""
    print("🔧 Executando testes unitários...")
    return subprocess.call([
        sys.executable, "-m", "pytest", 
        "-m", "unit", 
        "-v"
    ])


def run_integration_tests():
    """Executa apenas os testes de integração"""
    print("🔗 Executando testes de integração...")
    return subprocess.call([
        sys.executable, "-m", "pytest", 
        "-m", "integration", 
        "-v"
    ])


def run_coverage():
    """Executa testes com relatório de cobertura"""
    print("📈 Executando testes com cobertura de código...")
    return subprocess.call([
        sys.executable, "-m", "pytest", 
        "--cov=.", 
        "--cov-report=html", 
        "--cov-report=term-missing",
        "-v"
    ])


def run_specific_test(test_name):
    """Executa um teste específico"""
    print(f"🎯 Executando teste específico: {test_name}")
    return subprocess.call([
        sys.executable, "-m", "pytest", 
        "-v", 
        "-k", test_name
    ])


def main():
    parser = argparse.ArgumentParser(description="Executar testes do projeto")
    parser.add_argument(
        "tipo", 
        nargs="?", 
        default="all",
        choices=["all", "performance", "unit", "integration", "coverage"],
        help="Tipo de teste a executar"
    )
    parser.add_argument(
        "-k", "--test",
        help="Nome específico do teste a executar"
    )
    
    args = parser.parse_args()
    
    if args.test:
        exit_code = run_specific_test(args.test)
    elif args.tipo == "all":
        exit_code = run_all_tests()
    elif args.tipo == "performance":
        exit_code = run_performance_tests()
    elif args.tipo == "unit":
        exit_code = run_unit_tests()
    elif args.tipo == "integration":
        exit_code = run_integration_tests()
    elif args.tipo == "coverage":
        exit_code = run_coverage()
    
    if exit_code == 0:
        print("\n✅ Testes executados com sucesso!")
    else:
        print("\n❌ Alguns testes falharam!")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 