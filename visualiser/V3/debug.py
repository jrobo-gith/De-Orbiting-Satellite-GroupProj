can_print  = {"simulator": False,
           "predictor": False,
           "visualiser": False
}

def debug_print(module_name, print_str):
    """
    Print debug information if the module is set to print.
    :param module_name: Name of the module (simulator, predictor, visualiser)
    :param print_str: String to print
    """
    if can_print[module_name]:
        print(print_str)

dev_mode = False