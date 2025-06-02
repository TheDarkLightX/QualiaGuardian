# Part of a circular dependency test
from . import circ_a # Relative import

def func_b():
    print("Function B")
    # To avoid direct recursion during test, don't call func_a here in simple test
    # circ_a.func_a()

if __name__ == "__main__":
    # from guardian_ai_tool.dummy_project import circ_a as main_circ_a # for direct run
    func_b()
    # main_circ_a.func_a() # to ensure circ_a is "used"
    pass