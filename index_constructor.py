import json
import logging
import nltk
import psutil
import os
import time
import heapq

from typing import Generator, final
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
from collections import Counter, defaultdict, namedtuple, deque
from pprint import pprint


# nltk.download('stopwords')

logger = logging.getLogger(__name__)

class EmptyIndexError(Exception):
    pass

Posting = namedtuple("Posting", "doc_id, freq, ind, tfidf")

class InvertedIndex(object):

    webpages_normal_dir = "WEBPAGES_RAW/"
    webpages_small_dir = "WEBPAGES_RAW_SMALL/"
    stop_words_path = "english_stopwords.txt"
    bookkeeping_path = "bookkeeping.json"
    temp_index_dir = "TEMP_INDEX/"
    final_index_path = "final_inverted_index.tsv"

    nltk_tokenizer = RegexpTokenizer(r'[0-9]*[a-zA-Z][a-zA_Z0-9_]{2,30}')
    nltk_stemmer = SnowballStemmer("english")


    def __init__(self, debug: bool = False, use_small: bool = False) -> None:

        self.debug = debug

        # large or small corpus
        self.webpages_dir = self.webpages_small_dir if use_small\
            else self.webpages_normal_dir
        # parse bookkeeper json as dict
        book_path = os.path.join(self.webpages_dir, self.bookkeeping_path)
        with open(book_path) as f:
            self.bookkeeper_dict = json.load(f)
        # save stop words as set
        with open(self.stop_words_path, 'r') as f:
            self.stop_words = {w.strip() for w in f}


    def build_index(self):

        start = time.time()


        # if os.listdir(self.temp_index_dir):
        #     resp = None
        #     while resp.lower() not in ["yes", "no"]:
        #         resp = input("Do you want to recreate the index? (Yes|No)\n>>> ")
        #     if resp.lower() == "yes":



        # Creating blocks of inverted indexes

        # remove all files in temp index and temp merged index
        if not os.path.exists(self.temp_index_dir):
            os.mkdir(self.temp_index_dir)
        for f in os.listdir(self.temp_index_dir):
            os.remove(os.path.join(self.temp_index_dir, f))
        
        token_stream = self.token_generator()
        try:
            i = 0
            while True:
                temp_index_fname = os.path.join(self.temp_index_dir, "index_block_"+str(i)+".tsv")
                self.spimi_invert(temp_index_fname, token_stream)
                if self.debug:
                    logger.debug(f"successfully written: {temp_index_fname}")
                i += 1
        except EmptyIndexError:
            if self.debug:
                logger.debug("Empty index. Inverted index blocks are finished. Starting merge...")

        # Merging blocks of inverted indexes
        self.merge_indexes(self.final_index_path)
        if self.debug:
                logger.debug(f"Merging of index blocks complete. Created: {self.final_index_path}")

            

        end = time.time()
        if self.debug:
            logger.debug(f"Time elapsed to construct inverted index: {end-start}")

    def spimi_invert(self, fname, token_stream) -> None:
        """
        dictionary = newhash
        while (memory available):
            dictionary[token] += posting
        sort dictionary on tokens
        write to output_file
        return output_file(str)
        """
        empty_index = False
        index_block = defaultdict(deque)
        try:
            for _ in range(10000):  # testing purpose!!!
            # while (psutil.virtual_memory().percent < 90):
                token, posting = next(token_stream)
                index_block[token].append(posting)
        except StopIteration:
            empty_index = True
            if self.debug:
                logger.debug(f"No more tokens to generate! memory%: {psutil.virtual_memory().percent}")
            
        finally:
            with open(fname, 'w') as output:
                for token, posting in sorted(index_block.items()):
                    output.write(f"{token}\t{posting}\n")
            if empty_index:
                raise EmptyIndexError

    def token_generator(self) -> Generator:

        for file_path in self.bookkeeper_dict:
            with open(os.path.join(self.webpages_dir, file_path), encoding='utf8') as f:
                soup_text = BeautifulSoup(f.read(), "lxml").get_text(" ")

            # tokenization
            # FIXME might have issues not reading from headers and other tags
            tokens = self.nltk_tokenizer.tokenize(soup_text)

            # lemmatization and remove stop words (converts to generator)
            tokens = (self.nltk_stemmer.stem(t)
                      for t in tokens if t not in self.stop_words)

            # count frequencies and yield
            for token, freq in Counter(tokens).items():
                yield (token, Posting(file_path, freq, NotImplemented, NotImplemented))

            if self.debug:
                logger.debug(f"Tokenized file: {file_path:16}")


    def merge_indexes(self, final_index_path):
        
        # opening final index and all of the inverted index blocks
        final_index = open(final_index_path, 'w')
        file_objects = []
        for index_fname in os.listdir(self.temp_index_dir):
            file_objects.append(open(os.path.join(self.temp_index_dir, index_fname), 'r'))

        # initalize prioity queue with first entries of each file
        hq = []
        token_position = [] # token given by index
        for index_file in file_objects:
            token, posting_deque = index_file.readline().split("\t")
            hq.append((token, posting_deque))
            token_position.append(token + "\t" + posting_deque)
        
        # first token posting
        heapq.heapify(hq)
        min_token, min_postings = heapq.heappop(hq)
        curr_token, curr_dq = min_token, eval(min_postings)
        token_index = token_position.index(min_token + "\t" + min_postings)

        while hq:

            entity = file_objects[token_index].readline()
            token_position[token_index] = entity

            # if line is not a empty string, add to priority queue
            if entity:
                token, posting_deque = entity.split("\t")
                heapq.heappush(hq, (token, posting_deque))

            min_token, min_postings = heapq.heappop(hq) # checking next token
            token_index = token_position.index(min_token + "\t" + min_postings)

            # if new min token is same as current min token, 
            # merge the deques of postings.
            if curr_token == min_token: # previous token
                curr_dq = self.merge_sorted_lists(eval(min_postings), curr_dq)

            # if tokens are not the same as current min token,
            # write current(or prev) token to file
            elif curr_token != min_token:
                final_index.write(f"{curr_token}\t{curr_dq}\n")
                curr_token = min_token # next token
                curr_dq = eval(min_postings) # next curr_dq
        
        # last one        
        final_index.write(f"{curr_token}\t{curr_dq}\n")

    


    @staticmethod
    def merge_sorted_lists(l1: deque, l2: deque):

        new_list = deque()
        while True:
            if not l1:
                new_list.extend(l2)
                break
            elif not l2:
                new_list.extend(l1)
                break
            if l1[0] < l2[0]:
                new_list.append(l1.popleft())
            else:
                new_list.append(l2.popleft())
        return new_list