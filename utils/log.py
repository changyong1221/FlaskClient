import globals.global_var as glo


def print_log(str):
    if glo.is_print_log:
        print(str)
