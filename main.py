from datetime import *
from src import *


TRAIN_PATH = './assets/iris/train/'
TEST_PATH = './assets/iris/test/'
QUERY_IMG = f'{TEST_PATH}/001/L/S2001L05.jpg'
MAX_DISTANCE = 1


start = datetime.now()
timestamp = datetime.now()

train_set = IrisDataset(TRAIN_PATH)
test_set = IrisDataset(TEST_PATH)
matcher = IrisMatcher(train_set, num_tables=16)

print('Building CASIA gallery from train split...')
matcher.build_index()
print(f'Index build duration: {(datetime.now() - timestamp).total_seconds():.2f}s')
timestamp = datetime.now()

print(f'\nSearching matches for {QUERY_IMG}...')
results = matcher.search(QUERY_IMG, max_distance=MAX_DISTANCE)
# results = matcher.search(QUERY_IMG, tolerance_pct=0.15)
print(f'Search duration: {(datetime.now() - timestamp).total_seconds():.2f}s')
timestamp = datetime.now()

if results:
    print(f'\nPossible matches ({len(results)}):')
    for path, distance, descriptor in results[:min(len(results), 40)]:
        print(f' - {path} (Hamming distance={distance})')
else:
    print('No match.')

evaluator = Evaluator(matcher, test_set)
evaluator.evaluate(max_distance=MAX_DISTANCE)
print(f'\nEvaluation duration: {(datetime.now() - timestamp).total_seconds():.2f}s')

print(f'\nTotal duration: {(datetime.now() - start).total_seconds():.2f}s')
