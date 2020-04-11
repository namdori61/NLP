import jsonlines
from absl import app, flags, logging
from tqdm import tqdm
from khaiii import KhaiiiApi
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma

FLAGS = flags.FLAGS

flags.DEFINE_string('input', default=None, help='Path to input jsonl file')
flags.DEFINE_string('output', default=None, help='Path to save output')
flags.DEFINE_enum('method', default=None, enum_values=['komoran', 'okt', 'mecab', 'hannanum', 'kkma', 'khaiii'],
                  help='Method to tokenize')

def main(argv):
    input_path = FLAGS.input
    output_path = FLAGS.output

    if FLAGS.method == 'komoran':
        tokenizer = Komoran()
    elif FLAGS.method == 'okt':
        tokenizer = Okt()
    elif FLAGS.method == 'mecab':
        tokenizer = Mecab()
    elif FLAGS.method == 'hannanum':
        tokenizer = Hannanum()
    elif FLAGS.method == 'kkma':
        tokenizer = Kkma()
    elif FLAGS.method == 'khaiii':
        tokenizer = KhaiiiApi()
    else:
        raise ValueError('Unknown tokenize method')

    input_reader = jsonlines.open(input_path, 'r')
    output_writer = jsonlines.open(output_path, 'w')
    num_lines = 0
    for input_obj in tqdm(input_reader, desc='Processing'):
        messages = input_obj['messages']
        for message in messages:
            text = message['text']
            if FLAGS.method == 'khaiii':
                tokens = []
                for word in tokenizer.analyze(text):
                    tokens.extend([str(m).split('/')[0] for m in word.morphs])
            else:
                tokens = tokenizer.morphs(text)
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