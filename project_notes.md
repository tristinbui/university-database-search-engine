
### M1 requirements:
- **Inverted index**
    1. Tokenize each document/file
        - Using `nltk.tokenize.RegexpTokenizer` to tokenize only alphanumerics `[a-zA-Z0-9]{3,}`
    2. Linguistic modules to standardize tokens
        - Lemmatization on tokens
            - Using `nltk.stem.SnowballStemmer` stemmer (written by Martin Porter, aka Porter2 as it has slightly faster computation time)
        - Remove stop-words
    3. Build index
        - generate token stream of (token, posting) pairs
        - feed token stream into **SPIMI** inverted index constructor
        - merge index blocks

- **Retrieval system**

### Resources:
- Deliverable gdoc: https://drive.google.com/file/d/1a2EG1UNbYRyfHZtyvVc5AJPsOCTO9VGU/view
- Overview of M1: https://docs.google.com/presentation/d/1tVLM9IsTR05c53dN6japMkPwX6zYccI-_Bdy1XD8hsA/edit#slide=id.p
- SPIMI youtube video: https://www.youtube.com/watch?v=uXq4aq51eKE


### inverted index:
- key/token: the token (string)
- value/posting: 
    - [x] document id: relative path (string)
    - [x] word frequency (int)
    - [ ] indeces (tuple(int))
    - [ ] tf-idf score (float?)

