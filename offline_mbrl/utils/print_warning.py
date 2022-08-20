def print_warning(text, args):
    print(("{}" + text + "{}").format("\033[93m", *args, "\033[0m"))
