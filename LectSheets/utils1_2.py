def get_train_test_data(path='./scores.txt', train_size=0.8):
    
    data = []
    
    with open(path, 'r') as f: # f is only available within the code block

        for line in f:

            line_entries = line.strip().split(',')

            data.extend([float(x) for x in line_entries])

    print(f'Data length: {len(data)}')
    #f'File content: {data}'

    N = len(data)

    ratio = train_size
    split = int(ratio*N) # 80 % of length

    train_data = data[:split]
    test_data  = data[split:]

    print(f"Train len: {len(train_data)} Test len: {len(test_data)}")
    
    return train_data, test_data

def normalize():
    pass

def func(x):
    return x