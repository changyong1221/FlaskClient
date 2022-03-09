

def read_line_elem_from_file(file_path):
    """Read line data from given file (one line one elem)

    file_format:    1
                    2
                    3
                    4
    output_format:  [1, 2, 3, 4]

    """
    data_list = []
    with open(file_path, 'r') as f:
        for elem in f:
            data_list.append(int(elem))
        f.close()
    return data_list


def read_line_elems_from_file(file_path, delimiter=' '):
    """Read line data from given file (one line multiple elems)

    file_format:    1, 2, 3, 4
                    2, 3, 4, 5
    output_format:  [[1, 2, 3, 4]
                     [2, 3, 4, 5]]

    """
    data_vector = []
    with open(file_path, 'r') as f:
        for line in f:
            task = [float(val) for val in line.rstrip().split(delimiter)]
            data_vector.append(task)
        f.close()
    return data_vector


if __name__ == '__main__':
    file_path = "../dataset/GoCJ/Original_DataSet.txt"
    data = read_line_elem_from_file(file_path)
    print(len(data))
    print(data)
