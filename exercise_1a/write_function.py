def write_fn(output, file_path):
    with open(file_path,'a') as f:
        f.write(output+'\n')

if __name__ == '__main__':
    write_fn('testing', 'exercise_1a/test.txt')