# -* encoding: utf-8 *-

def main(devices = -1):
    pass

if __name__ == '__main__':
    import sys

    use_device = -1
    if len(sys.argv) >= 2:
        use_device = sys.argv[1]
    main(use_device)