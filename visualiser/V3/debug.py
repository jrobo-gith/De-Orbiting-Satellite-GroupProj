can_print  = {"simulator": False,
           "predictor": False,
           "visualiser": False
}

def debug_print(module_name, print_str):
    if can_print[module_name]:
        print(print_str)

dev_mode = False