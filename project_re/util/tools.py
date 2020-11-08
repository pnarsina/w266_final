from __future__ import absolute_import, division, print_function

import csv
import os
import sys
import logging
import json

logger = logging.getLogger()
csv.field_size_limit(2147483647) # Increase CSV reader's field limit incase we have long text.

CONFIG_FOLDER = 'config/'
id_label_file = 'id_2_label.json'
lable_2_id_file = 'label2_2_id.json'
train_file = 'data/tacred/data/json/train.json'


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
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


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

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

    def get_labels(self):
        """See base class."""
        return ['no_relation' , 'org:subsidiaries' , 'org:city_of_headquarters' , 'per:title',
             'per:origin' , 'per:employee_of' , 'org:top_members/employees',
             'org:alternate_names' , 'org:shareholders' , 'org:country_of_headquarters',
             'per:countries_of_residence' , 'per:date_of_death',
             'per:cities_of_residence' , 'per:city_of_death' , 'per:age' , 'org:founded_by',
             'org:parents' , 'org:member_of' , 'per:stateorprovinces_of_residence',
             'per:religion' , 'org:founded' , 'org:stateorprovince_of_headquarters',
             'per:alternate_names' , 'per:siblings' , 'per:charges',
             'org:number_of_employees/members' , 'per:stateorprovince_of_death',
             'org:members' , 'per:cause_of_death' , 'per:parents' , 'per:other_family',
             'per:schools_attended' , 'per:children' , 'per:spouse' , 'per:country_of_birth',
             'org:political/religious_affiliation' , 'per:country_of_death',
             'per:date_of_birth' , 'per:city_of_birth' , 'org:website' , 'org:dissolved',
             'per:stateorprovince_of_birth']

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
