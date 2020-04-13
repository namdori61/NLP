import jsonlines
from absl import app, flags, logging
from tqdm import tqdm
from soyspacing.countbase import CountSpace
import math
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer
import sentencepiece as spm

FLAGS = flags.FLAGS

flags.DEFINE_string('input', default=None, help='Path to input jsonl file')
flags.DEFINE_string('output', default=None, help='Path to save output')
flags.DEFINE_enum('method', default=None, enum_values=['soynlp', 'sentencepiece'], help='Method to tokenize')
flags.DEFINE_boolean('spacing', default=False, help='(Optional) Whether to correct spacing before tokenizing')
flags.DEFINE_string('model', default=None, help='(Optional) Path to model input')
flags.DEFINE_integer('vocab_size', default=100000, help='(Optional) Output vocabulary size of sentencepiece tokenizer')

def make_corpus(input_path, output_path):
    input_reader = jsonlines.open(input_path, 'r')
    with open(output_path,'w') as f:
        for input_obj in tqdm(input_reader, desc='Processing Corpus'):
            messages = input_obj['messages']
            for message in messages:
                text = message['text']
                f.write(text + '\n')

def train_soy_spacing(input_path, model_path):
    model = CountSpace()
    model.train(input_path)
    model.save_model(model_path, json_format=False)

def apply_soy_spacing(input_path, model_path, output_path):
    model = CountSpace()
    model.load_model(model_path, json_format=False)
    with open(input_path, 'r', encoding='utf-8') as f1, \
        open(output_path, 'w', encoding='utf-8') as f2:
        for sentence in f1:
            sentence = sentence.strip()
            if not sentence: continue
            sent_corrected, _ = model.correct(sentence)
            f2.writelines(sent_corrected + "\n")

def train_tokenizer(input_path, method, vocab_size):
    if method == 'soynlp':
        sentences = [sent.strip() for sent in open(input_path, 'r').readlines()]
        word_extractor = WordExtractor(min_frequency=100,
                                    min_cohesion_forward=0.05,
                                    min_right_branching_entropy=0.0
                                    )
        word_extractor.train(sentences)
        word_extractor.save(method+'.model')
    elif method == 'sentencepiece':
        train='--input=' + input_path + ' --model_prefix=' + method + ' --vocab_size=' + str(vocab_size) + ' --model_type=bpe --character_coverage=0.9995'
        spm.SentencePieceTrainer.train(train)
    else:
        raise ValueError('Unknown tokenize method')

def init_tokenizer(method, model_path):
    if method == 'soynlp':
        word_extractor = WordExtractor(min_frequency=100,
                                   min_cohesion_forward=0.05,
                                   min_right_branching_entropy=0.0
                                   )
        word_extractor.load(model_path)
        scores = word_extractor.word_scores()
        words = word_extractor.extract()
        scores = {key:(scores[key].cohesion_forward * math.exp(scores[key].right_branching_entropy)) for key in scores.keys()}
        tokenizer = MaxScoreTokenizer(scores=scores)
    elif method == 'sentencepiece':
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(model_path)
    else:
        raise ValueError('Unknown tokenize method')
    return tokenizer

def main(argv):
    input_path = FLAGS.input
    output_path = FLAGS.output
    spacing = FLAGS.spacing
    method = FLAGS.method
    model_path = FLAGS.model
    vocab_size = FLAGS.vocab_size

    if model_path is None:
        logging.info('Start training model...')

        logging.info('Making corpus...')
        make_corpus(input_path, 'corpus.txt')

        if spacing == True:
            logging.info('Spacing corpus...')
            train_soy_spacing('corpus.txt', 'soy_spacing.model')
            apply_soy_spacing('corpus.txt', 'soy_spacing.model', 'spaced_corpus.txt')

            logging.info(f'Training {method} tokenizer...')
            train_tokenizer('spaced_corpus.txt', method, vocab_size)
            tokenizer = init_tokenizer(method, method+'.model')
        else:
            logging.info(f'Training {method} tokenizer...')
            train_tokenizer('corpus.txt', method, vocab_size)
            tokenizer = init_tokenizer(method, method+'.model')
    else:
        tokenizer = init_tokenizer(method, model_path)

    input_reader = jsonlines.open(input_path, 'r')
    output_writer = jsonlines.open(output_path, 'w')
    num_lines = 0
    for input_obj in tqdm(input_reader, desc='Processing tokenize'):
        messages = input_obj['messages']
        for message in messages:
            text = message['text']
            if method == 'soynlp':
                tokens = tokenizer.tokenize(text)
            else:
                tokens = tokenizer.encode_as_pieces(text)
                tokens = [token.replace('▁','') for token in tokens]
            tokenized_text = ' '.join(tokens)
            message['text'] = tokenized_text
        output_writer.write(input_obj)
        num_lines += 1
    input_reader.close()
    output_writer.close()
    logging.info(f'Processed {num_lines} lines')

if __name__ == '__main__':
    flags.mark_flags_as_required(['input', 'output', 'method'])
    app.run(main)