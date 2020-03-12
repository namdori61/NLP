import re, json, glob, argparse
from gensim.corpora import WikiCorpus, Dictionary
from gensim.utils import to_unicode

"""
Preprocess a corpus from kowiki, korquad, nsmc, and namuwiki dump file.
Code to preprocess kowiki, korquad, and nsmc dump file is inspired by Ratsgo:
https://github.com/ratsgo/embedding
"""

def process_wiki(in_f, out_f):
    """Convert Wikipedia xml dump file to text corpus"""
    output = open(out_f, 'w')
    wiki = WikiCorpus(in_f, tokenizer_func=tokenize_wiki, dictionary=Dictionary())
    i = 0
    for text in wiki.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        i = i + 1
        if (i % 10000 == 0):
            print('Processed ' + str(i) + ' articles')
    output.close()
    print('Processing complete!')

WIKI_REMOVE_CHARS = re.compile("'+|(=+.{2,30}=+)|__TOC__|(ファイル:).+|:(en|de|it|fr|es|kr|zh|no|fi):|\n", re.UNICODE)
WIKI_SPACE_CHARS = re.compile("(\\s|゙|゚|　)+", re.UNICODE)
EMAIL_PATTERN = re.compile("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", re.UNICODE)
URL_PATTERN = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.UNICODE)
WIKI_REMOVE_TOKEN_CHARS = re.compile("(\\*$|:$|^파일:.+|^;)", re.UNICODE)
MULTIPLE_SPACES = re.compile(' +', re.UNICODE)

def tokenize_wiki(content, token_min_len=2, token_max_len=100, lower=True):
    content = re.sub(EMAIL_PATTERN, ' ', content)  # remove email pattern
    content = re.sub(URL_PATTERN, ' ', content) # remove url pattern
    content = re.sub(WIKI_REMOVE_CHARS, ' ', content)  # remove unnecessary chars
    content = re.sub(WIKI_SPACE_CHARS, ' ', content)
    content = re.sub(MULTIPLE_SPACES, ' ', content)
    tokens = content.replace(", )", "").split(" ")
    result = []
    for token in tokens:
        if not token.startswith('_'):
            token_candidate = to_unicode(re.sub(WIKI_REMOVE_TOKEN_CHARS, '', token))
        else:
            token_candidate = ""
        if len(token_candidate) > 0:
            result.append(token_candidate)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, help='Corpus to preprocess')
    parser.add_argument('--input_path', type=str, help='Location of input files')
    parser.add_argument('--output_path', type=str, help='Location of output files')
    parser.add_argument('--with_label', help='with label', type=str, default="False")
    args = parser.parse_args()

    if args.corpus == "wiki":
        process_wiki(args.input_path, args.output_path)
    elif "nsmc" in args.corpus:
        process_nsmc(args.input_path, args.output_path, "json" in args.corpus, args.with_label.lower() == "true")
    elif args.corpus == "korquad":
        process_korQuAD(args.input_path, args.output_path)
    elif args.corpus == "process-documents":
        process_documents(args.input_path, args.output_path)