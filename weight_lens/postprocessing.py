import re
import nltk
import difflib
from nltk.stem import WordNetLemmatizer

system_prompt = """
            We're studying neurons in a neural network. Each neuron has certain inputs that activate it and outputs that it leads to. You will receive three pieces of information about a neuron: 

            1. The top important tokens.
            2. The top tokens it promotes in the output. 
            3. The tokens it suppresses in the output.

            These will be separated into three sections [Important Tokens] and [Text Promoted] and [Text Suppressed]. All three are a combination of tokens. You can infer the most likely output or function of the neuron based on these tokens. The tokens, specially [Text Promoted] and [Text Suppressed] may include noise, such as unrelated terms, symbols, or programming jargon. If these are not coherent, you may ignore them and do not include them in your response. If the [Important Tokens] are not combining to form a common theme, you may simply combine the words in the [Important Tokens] to form a single concept.

            Focus on identifying a cohesive theme or concept shared by the most relevant tokens. 

            Your response should be a concise (1-2 sentence) explanation of the neuron, encompassing what triggers it (input) and what it does once triggered (output). If the two sides relate to one another you may include that in your explanation, otherwise simply state the input and output. Give your output in the following format:

            [Concept: <Your interpretation of the neuron, based on the tokens provided>]

            Example 1:

            Input:
            [Important Tokens]: ['accused', 'saw'] 
            [Tokens Promoted]: ['tvguidetime', 'hasfactory']
            [Tokens Suppressed]: ["'", '']

            Output:
            [Concept: The verbs "accused" and "saw"]

            Example 2:
            Input:
            [Important Tokens]: ['on', 'pada']
            [Tokens Promoted]: ['behalf']
            [Tokens Suppressed]: ['on', 'in']
            
            Output:
            [Concept: The token "on" in the context of "on behalf of" and so on]


            Example 3:
            Input:
            [Important Tokens]: ['carrier', 'missing', '']
            [Tokens Promoted]: None
            [Tokens Suppressed]: None

            Output:
            [Concept: the word "missing" and "carrier" ]

            Example 4:
            Input:
            [Important Tokens]: ['democratic', 'dare']
            [Tokens Promoted]: nan
            [Tokens Suppressed]: nan
            [Concept: The tokens "democratic" and "dare"]
            




            Important: Only output the [Concept] as your response. Do not include any other text or explanation for the same.
                    
            """

# Download NLTK resources needed for lemmatization
nltk.download("wordnet")
nltk.download("omw-1.4")

def is_punctuation_like(token):
    """
    Check if a token consists entirely of punctuation characters.

    Args:
        token (str): The token to check.

    Returns:
        bool: True if all characters in the token are punctuation, False otherwise.
    """
    return all(c in r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~" for c in token)

def normalize_token(token):
    """
    Normalize a token by lowercasing, stripping whitespace, and removing punctuation
    (except if the token is purely punctuation).

    Args:
        token (str): The token to normalize.

    Returns:
        str: The normalized token.
    """
    token = token.strip().lower()
    if is_punctuation_like(token):
        # Leave pure punctuation tokens unchanged
        return token
    # Remove punctuation characters from the token
    token = re.sub(r"[^\w\s]", "", token)
    return token

def group_similar(tokens, threshold=0.6):
    """
    Group similar tokens together based on sequence similarity.

    This function uses a greedy clustering approach: each token is compared to
    existing groups, and if it matches one above the threshold, it is added to that group.
    Otherwise, a new group is started.

    Args:
        tokens (list of str): List of tokens to group.
        threshold (float): Similarity ratio threshold (0–1) for grouping.

    Returns:
        list of str: Representative token (first token) from each group.
    """
    groups = []
    for token in tokens:
        found_group = False
        for group in groups:
            # Compare with each member of the group
            if any(difflib.SequenceMatcher(None, token, member).ratio() > threshold for member in group):
                group.append(token)
                found_group = True
                break
        if not found_group:
            groups.append([token])
    # Return the first token from each group as representative
    return [group[0] for group in groups]

def unique_tokens(tokens, word_threshold=0.85, punct_threshold=0.6):
    """
    Extract a list of unique tokens from an input list, accounting for case, punctuation,
    lemmatization, and similarity grouping.

    - Words are lowercased, stripped, and lemmatized before similarity grouping.
    - Punctuation tokens are grouped separately with a different similarity threshold.
    - Both sets are merged into a final unique list.

    Args:
        tokens (list of str): List of original tokens.
        word_threshold (float): Similarity threshold for grouping words.
        punct_threshold (float): Similarity threshold for grouping punctuation tokens.

    Returns:
        list of str: Unique representative tokens after processing.
    """
    lemmatizer = WordNetLemmatizer()
    normalized = [normalize_token(tok) for tok in tokens]

    words = []
    punct = []

    # Separate words from punctuation-like tokens
    for tok in normalized:
        if is_punctuation_like(tok):
            punct.append(tok)
        else:
            lemmatized = lemmatizer.lemmatize(tok)
            words.append(lemmatized)

    # Group similar words and punctuation separately
    unique_words = group_similar(words, threshold=word_threshold)
    unique_punct = group_similar(punct, threshold=punct_threshold)

    # Merge the results
    un_tokens = unique_words + unique_punct

    # If all tokens were empty strings, return original tokens
    if un_tokens == ['']:
        return tokens

    return un_tokens
