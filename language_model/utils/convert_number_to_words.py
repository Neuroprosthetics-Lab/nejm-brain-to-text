import re
import time

from tqdm import tqdm
from num2words import num2words

from multiprocessing import Pool, Queue

def buf_count_newlines_gen(fname):
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count

def number_to_words(line:str):
    def transform(match):
        number_str = match.group()
        # print('\t' + number_str, end=' ')

        number_str = re.sub(r'\.+', '.', number_str)
        if number_str.endswith('.'):
            number_str = number_str[:-1]

        # check if number is a percentage
        if '%' in number_str: 
            new_number_str = num2words(re.sub('\.?[$%\b]+', '', number_str)) + ' percent'
        # check if number is a valid year
        elif re.match(r'\b^\d{4}\b', number_str) and 1800 <= int(re.sub('[^\d]+', '', number_str)) <= 2100:
            new_number_str =  num2words(re.sub('[^\d]+', '', number_str), to='year')
        # check if multiple '.' in number (e.g. 3.1.1)
        elif number_str.count('.') > 0:

            new_number_str =  ' point '.join([num2words(re.sub('[^\d]+', '', num)) for num in number_str.split('.')])
        else:
            new_number_str =  num2words(re.sub('[^\d.]+', ' ', number_str))
        # print(new_number_str)
        return ' ' + new_number_str + ' '
    
    new_line = re.sub(r'\$?[\d]+[\d\.]*%?', transform, line)
    new_line = re.sub(r'\s+', ' ', new_line)
    return new_line.strip()

def process_line(input_queue:Queue, output_queue:Queue, error_queue:Queue):
    while True:
        line = input_queue.get(True)

        line = line.strip()[:-1]
        if '...' in line:
            error_queue.put(line)
            continue

        try:
            # Replace numbers with words
            new_line = number_to_words(line)
            new_line = re.sub(r'[^a-zA-z0-9\' ]', '', new_line)
            new_line = re.sub(r'\s+', ' ', new_line)
            output_queue.put(new_line)
        except:
            error_queue.put(line)

def write_queue_to_file(queue:Queue, file:str, wait_for_queue:Queue):
    while not wait_for_queue.empty():
        time.sleep(1)
    with open(file, 'a') as f:
        while not queue.empty():
            f.write(queue.get() + '\n')


if __name__ == '__main__':
    SOURCE_FILE = 'financial-reports-sec.txt'
    OUTPUT_FILE = 'financial-reports-sec_processed.txt'
    ERROR_FILE = 'financial-reports-sec_error.txt'

    total_line_count = buf_count_newlines_gen(SOURCE_FILE)

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    pool = Pool(20, process_line, (input_queue, output_queue, error_queue,))

    start_time = time.time()

    # Clear files
    open(OUTPUT_FILE, 'w').close()
    open(ERROR_FILE, 'w').close()

    pb = tqdm(total=total_line_count)
    with open(SOURCE_FILE, 'r') as fp:
        for i, line in enumerate(fp):
            input_queue.put(line)
        
            if i % 100 == 0 and i != 0:
                pb.update(100)
    pb.update(total_line_count % 100)
    pb.close()
    
    print('Finished reading file, processing...')
    pb = tqdm(total=total_line_count)
    while not input_queue.empty():
        time.sleep(1)
        pb.update(output_queue.qsize() + error_queue.qsize() - pb.n)
    pb.update(output_queue.qsize() + error_queue.qsize() - pb.n)
    pb.close()

    print('Finished processing file, writing to file...')
    write_queue_to_file(output_queue, OUTPUT_FILE, input_queue)
    write_queue_to_file(error_queue, ERROR_FILE, input_queue)

    pool.close()