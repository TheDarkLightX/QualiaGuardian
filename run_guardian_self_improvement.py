from guardian.self_improvement.guardian_optimizer import GuardianOptimizer, SelectionMode

def main():
    """
    Runs the Guardian self-improvement process.
    """
    optimizer = GuardianOptimizer(
        guardian_root="guardian_ai_tool/guardian",  # Path relative to workspace root
        selection_mode=SelectionMode.GUIDED
    )

    print("Starting Guardian self-improvement cycle...")
    results = optimizer.run_improvement_cycle(max_iterations=20)
    print("\nGuardian self-improvement cycle finished.")
    print("Results:")
    print(results)

if __name__ == "__main__":
    main()