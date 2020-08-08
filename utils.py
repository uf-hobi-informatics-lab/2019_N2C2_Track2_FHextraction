import csv
import os
import sys
import pickle as pkl

from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

from models import BertForEntityClassification, BertForRelationIdentification


def load_text(ifn):
    with open(ifn, "r") as f:
        txt = f.read()
    return txt


def save_text(text, ofn):
    with open(ofn, "w") as f:
        f.write(text)


def pkl_save(data, file):
    with open(file, "wb") as f:
        pkl.dump(data, f)


def pkl_load(file):
    with open(file, "rb") as f:
        data = pkl.load(f)
    return data


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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    try:
        ## out the positive class as the first label, so pos_label=0
        f1 = f1_score(y_true=labels, y_pred=preds, pos_label=0, average='binary')
    except:
        f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    
    p, r, f, s = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='weighted')
    return {
        "acc": acc,
        "f1": f1,
        "F1": f,
        "Pre": p,
        "Rec": r
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in {"fms", "fmr", "obn", "lss", "rel1", "rel2"}:
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


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
                    line = list(cell for cell in line)
                lines.append(line)
            return lines


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

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
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class FamilyHistoryRelationPlan1(Sst2Processor):
    def get_labels(self):
        """See base class."""
        return ["pos", "neg"]

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            if set_type == "test":
                label = self.get_labels()[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class FamilyHistoryRelationPlan2(Sst2Processor):
    def get_labels(self):
        """See base class."""
        return ["pos", "neg"]

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            if set_type == "test":
                label = self.get_labels()[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class FamilySideProcessor(Sst2Processor):
    def get_labels(self):
        """See base class."""
        return ["NA", "Maternal", "Paternal"]

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            if set_type == "test":
                label = self.get_labels()[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class FamilyRoleProcessor(Sst2Processor):
    def get_labels(self):
        """See base class."""
        return ['Parent', 'Mother', 'Grandparent', 'Sister', 'Brother', 'Child', 'Cousin', 'Sibling', 
        'Uncle', 'Father', 'Son', 'Grandmother', 'Aunt', 'Daughter', 'Grandfather']

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            if set_type == "test":
                label = self.get_labels()[0]
            else:
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ObNegationProcessor(FamilySideProcessor):
    def get_labels(self):
        """See base class."""
        return ["Non_Negated", "Negated"]

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            if set_type == "test":
                label = self.get_labels()[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class LivingStatusProcessor(ObNegationProcessor):
    def get_labels(self):
        """See base class."""
        return ["0", "2", "4"]


processors = {
        "fms": FamilySideProcessor,
        "fmr": FamilyRoleProcessor,
        "obn": ObNegationProcessor,
        "lss": LivingStatusProcessor,
        "rel": FamilyHistoryRelationPlan2,
        "rel1": FamilyHistoryRelationPlan1,
        "rel2": FamilyHistoryRelationPlan2
    }


output_modes = {
        "fms": "classification",
        "fmr": "classification",
        "obn": "classification",
        "lss": "classification",
        "rel": "classification",
        "rel1": "classification",
        "rel2": "classification"
    }


MODEL_CLASSES = {
    # customized classifier using both [CLS] and extra tag
    "berta": (BertConfig, BertForEntityClassification, BertTokenizer),
    "bertr": (BertConfig, BertForRelationIdentification, BertTokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
}
