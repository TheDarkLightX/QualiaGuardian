# Part of a circular dependency test
from . import circ_b # Relative import

def func_a():
    print("Function A")
    circ_b.func_b()

if __name__ == "__main__":
    # To run this directly, Python needs to treat dummy_project as a package
    # This might require running with python -m dummy_project.circ_a
    # For Guardian's analysis, the import structure is what matters.
    # from guardian_ai_tool.dummy_project import circ_b as main_circ_b # for direct run
    # main_circ_b.func_b() # to ensure circ_b is "used"
    pass