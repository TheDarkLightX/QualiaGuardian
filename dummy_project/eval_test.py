# This file is for testing eval() detection.

def use_eval_safely_maybe(data):
    # This is a comment, eval("ignored")
    print("About to use eval...")
    return eval(data) # This should be detected

def another_function():
    x = "print('Hello from eval in another_function')"
    eval (x) # This should also be detected
    not_an_eval = "evaluation_strategy"

# eval() in a string literal should not be detected by basic regex
string_with_eval = "this is a string with eval() in it"

# commented_out_eval = eval("this is commented out")