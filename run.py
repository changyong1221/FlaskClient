import os
import argparse
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(100)


def create_app(paras):
    port = paras["port"]
    id = paras["id"]
    os.system(f"python app.py -p {port} -i {id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', dest="number", type=int, default=1)
    args = parser.parse_args()
    clients_num = args.number
    if clients_num <= 0:
        print("Error: parameter n >= 1 !")
    start_port = 4000
    i = 1
    while i <= clients_num:
        port = start_port + i
        id = i - 1
        executor.submit(create_app, ({"port": port, "id": id}))
        i += 1
