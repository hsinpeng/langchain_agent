import sys
def main():
    try:
        print("Hello, World!")
    
        run_option = 0
        match run_option:
            case 0:
                print('Hello')
            case _:
                print(f'Error: Wrong run_option({run_option})!')

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())