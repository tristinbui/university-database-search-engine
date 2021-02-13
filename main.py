
from index_constructor import InvertedIndex
import logging

logging.basicConfig(format='%(asctime)s (%(name)s:%(funcName)s) %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

# index = InvertedIndex(debug=True, use_small=True)
index = InvertedIndex(debug=True, use_small=False)

index.build_index()