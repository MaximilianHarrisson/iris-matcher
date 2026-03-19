from datetime import datetime
from src import IrisMatcher


IRIS_PATH = './assets/iris/'
QUERY_IMG = './assets/query.jpg'


start = datetime.now()

matcher = IrisMatcher(IRIS_PATH, num_tables=4)
print('Building CASIA base indexes...')
matcher.build_index()
print(f'Index build duration: {(datetime.now() - start).total_seconds():.2f}s')

print(f'\nSearching matches for {QUERY_IMG}...')
search_start = datetime.now()
results = matcher.search(QUERY_IMG, max_distance=4)
# results = matcher.search(QUERY_IMG, tolerance_pct=0.15)
print(f'Search duration: {(datetime.now() - search_start).total_seconds():.2f}s')

if results:
    print(f'\nPossible matches ({len(results)}):')
    for path, distance, descriptor in results:
        print(f' - {path} (Hamming distance={distance}) (Descriptor={descriptor})')
else:
    print('No match.')

print(f'\nTotal duration: {(datetime.now() - start).total_seconds():.2f}s')
