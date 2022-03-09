

def write_list_to_file(data_list, file_path, mode='a+', delimiter='\t'):
    """Write a list object to specified file. A list is written in one line.

    list format: [1, 2.32, 423, 4.554, 5.3]
    file format: 1, 2.32, 423, 4.554, 5.3

    """
    with open(file_path, mode) as f:
        for elem in data_list[:-1]:
            f.write(str(elem) + delimiter)
        f.write(str(data_list[-1]))
        f.write('\n')
        f.close()


def write_vector_to_file(data_vector, file_path, mode='w', delimiter='\t'):
    """Write a vector to specified file

    vector format: [[1, 2, 3, 4, 5]
                    [6, 7, 8, 9, 1]
                    [2, 3, 4, 5, 6]]
    file_format:    1, 2, 3, 4, 5
                    6, 7, 8, 9, 1
                    2, 3, 4, 5, 6

    """
    with open(file_path, mode) as f:
        for lst in data_vector:
            for elem in lst[:-1]:
                f.write(str(elem) + delimiter)
            f.write(str(lst[-1]))
            f.write('\n')
        f.close()


def write_simple_list_to_file(simple_list, file_path, mode='w'):
    """Write a simple list to specified file. An list elem is written in one line.

    list format: [1, 2, 3, 4, 5]
    file_format:  1
                  2
                  3
                  4
                  5

    """
    with open(file_path, mode) as f:
        for elem in simple_list:
            f.write(str(elem) + '\n')
        f.close()
