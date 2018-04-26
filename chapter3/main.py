# -* encoding: utf-8 *-

def main(use_device=0):
    print("main chapter3")

if __name__ == '__main__':
    import sys

    use_device = 0

    if len(sys.argv) >= 2:
        use_device = sys.argv[1]

    maih(use_device)
