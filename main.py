from datetime import *
from src import *


IRIS_PATH = './assets/iris/'
QUERY_IMG = './assets/query.jpg'
MAX_DISTANCE = 20


start = datetime.now()
timestamp = datetime.now()

dataset = IrisDataset(IRIS_PATH)
matcher = IrisMatcher(dataset, num_tables=16)

print('Building CASIA base indexes...')
matcher.build_index()
print(f'Index build duration: {(datetime.now() - timestamp).total_seconds():.2f}s')
timestamp = datetime.now()

print(f'\nSearching matches for {QUERY_IMG}...')
results = matcher.search(QUERY_IMG, max_distance=1)
# results = matcher.search(QUERY_IMG, tolerance_pct=0.15)
print(f'Search duration: {(datetime.now() - timestamp).total_seconds():.2f}s')
timestamp = datetime.now()

if results:
    print(f'\nPossible matches ({len(results)}):')
    for path, distance, descriptor in results[:min(len(results), 40)]:
        print(f' - {path} (Hamming distance={distance})')
else:
    print('No match.')

evaluator = Evaluator(matcher)
evaluator.evaluate(max_distance=1)
print(f'\nEvaluation duration: {(datetime.now() - timestamp).total_seconds():.2f}s')

print(f'\nTotal duration: {(datetime.now() - start).total_seconds():.2f}s')
