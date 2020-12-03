# This python file helps to get the data from the files, format and make it ready for transformers
from .tools import *
from transformers import BertTokenizer
from multiprocessing import Pool, cpu_count
import pickle, copy
import logging
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm , trange
import torch


CONFIG_FOLDER = 'config/'
id_label_file = 'id_2_label.json'
lable_2_id_file = 'label2_2_id.json'


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_feature(example_row):
    # return example_row
    example,  max_seq_length, tokenizer  = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length


    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=example.label)
        
        
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        
        raise NotImplementedError()
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def createDirectories(cls,config):
        report_dir = config.programsettings.REPORTS_DIR
#         if os.path.exists(report_dir) and os.listdir(report_dir):
#             report_dir += f'/report_{len(os.listdir(report_dir))}'
#             os.makedirs(report_dir)

        output_dir = config.programsettings.OUTPUT_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

        

class MultiClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        
        return ['Reason-Drug', 'Route-Drug', 'Strength-Drug', 'Frequency-Drug',
       'Duration-Drug', 'Form-Drug', 'Dosage-Drug', 'ADE-Drug',
       'no relation']

#         return ['no_relation' , 'org:subsidiaries' , 'org:city_of_headquarters' , 'per:title',
#              'per:origin' , 'per:employee_of' , 'org:top_members/employees',
#              'org:alternate_names' , 'org:shareholders' , 'org:country_of_headquarters',
#              'per:countries_of_residence' , 'per:date_of_death',
#              'per:cities_of_residence' , 'per:city_of_death' , 'per:age' , 'org:founded_by',
#              'org:parents' , 'org:member_of' , 'per:stateorprovinces_of_residence',
#              'per:religion' , 'org:founded' , 'org:stateorprovince_of_headquarters',
#              'per:alternate_names' , 'per:siblings' , 'per:charges',
#              'org:number_of_employees/members' , 'per:stateorprovince_of_death',
#              'org:members' , 'per:cause_of_death' , 'per:parents' , 'per:other_family',
#              'per:schools_attended' , 'per:children' , 'per:spouse' , 'per:country_of_birth',
#              'org:political/religious_affiliation' , 'per:country_of_death',
#              'per:date_of_birth' , 'per:city_of_birth' , 'org:website' , 'org:dissolved',
#              'per:stateorprovince_of_birth']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
        
    def get_data_loader(self, config, source="train"):

        logging.basicConfig(level=logging.INFO)
    
        self.config = config
        
    #   Create output, report directories, if doesn't exist already
        self.createDirectories(config)
    #   This is to read input data and process them 

        if source == "train":
            data = self.get_train_examples(config.programsettings.DATA_DIR)
        elif source == "dev":
            data = self.get_dev_examples(config.programsettings.DATA_DIR)
        elif source == "test":
            data = self.get_test_examples(config.programsettings.DATA_DIR)

        data_len = len(data)

        label_list = self.get_labels() # [0, 1] for binary classification
        num_labels = len(label_list)
        num_train_optimization_steps = int(
        data_len / config.hyperparams.TRAIN_BATCH_SIZE / config.hyperparams.GRADIENT_ACCUMULATION_STEPS) * config.hyperparams.NUM_TRAIN_EPOCHS

        seq_length = str(config.hyperparams.MAX_SEQ_LENGTH)
        
        if source == "train":
            feature_pickle_file = config.programsettings.DATA_DIR + "train_features_" + seq_length + ".pkl"
        elif source == "dev":
            feature_pickle_file = config.programsettings.DATA_DIR + "dev_features_" + seq_length + ".pkl"
        elif source == "test":
            feature_pickle_file = config.programsettings.DATA_DIR + "test_features_" + seq_length + ".pkl"
        
        print("Looking for cached feature pickle file", feature_pickle_file)
        
        if not os.path.exists(feature_pickle_file):

            tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

            examples_for_processing = [(example,  config.hyperparams.MAX_SEQ_LENGTH, tokenizer) for example in data]

            process_count = cpu_count() - 1
            
            with Pool(process_count) as p:
                features = list(tqdm(p.imap(convert_example_to_feature, examples_for_processing), total=data_len))
                
            with open(feature_pickle_file, "wb") as f:
                pickle.dump(features, f)  
                
        with open(feature_pickle_file, "rb") as f:
            features = pickle.load(f)


        logger.info("  Num examples = %d", data_len)
        logger.info("  Batch size = %d", config.hyperparams.TRAIN_BATCH_SIZE)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([int(f.label_id) for f in features], dtype=torch.long)    

        tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        t_sampler = RandomSampler(tensor_data)
        dataloader = DataLoader(tensor_data, sampler=t_sampler, batch_size=config.hyperparams.TRAIN_BATCH_SIZE)

        return dataloader, data_len, num_labels, num_train_optimization_steps, all_label_ids