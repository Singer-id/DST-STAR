import csv
import os


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if len(line) > 0 and line[0][0] == '#':  # ignore comments (starting with '#')
                continue
            lines.append(line)
        return lines


def get_test_instances(data_dir):
    return _create_instances(_read_tsv(os.path.join(data_dir, "test.tsv")))


def _create_instances(ontology, lines):
    slot_meta = list(ontology.keys())  # must be sorted
    num_slots = len(slot_meta)
    slot_idx = [*range(0, num_slots)]  # *是解包的意思
    label_list = [ontology[slot] for slot in slot_meta]
    description = json.load(open("utils/slot_description.json", 'r'))
    slot_type = description[slot_meta]["value_type"]

    instances = []
    last_uttr = None
    last_dialogue_state = {}
    history_uttr = []

    for (i, line) in enumerate(lines):
        dialogue_idx = line[0]
        turn_idx = int(line[1])
        is_last_turn = (line[2] == "True")
        system_response = line[3]
        user_utterance = line[4]
        turn_dialogue_state = {}
        turn_dialogue_state_ids = []
        for idx in self.slot_idx:
            turn_dialogue_state[self.slot_meta[idx]] = line[5 + idx]
            turn_dialogue_state_ids.append(self.label_map[idx][line[5 + idx]])

        if turn_idx == 0:  # a new dialogue
            last_dialogue_state = {}
            history_uttr = []
            last_uttr = ""
            for slot in self.slot_meta:
                last_dialogue_state[slot] = "none"

        turn_only_label = []  # turn label
        for s, slot in enumerate(self.slot_meta):
            if last_dialogue_state[slot] != turn_dialogue_state[slot]:
                turn_only_label.append(slot + "-" + turn_dialogue_state[slot])


        history_uttr.append(last_uttr)

        text_a = (system_response + " " + user_utterance).strip()
        text_b = ' '.join(history_uttr[-self.config.num_history:])
        last_uttr = text_a

        # ID, turn_id, turn_utter, dialogue_history, label_ids,
        # turn_label, curr_turn_state, last_turn_state,
        # max_seq_length, slot_meta, is_last_turn, ontology
        instance = TrainingInstance(dialogue_idx, turn_idx, text_a + " none ", text_b, turn_dialogue_state_ids,
                                    turn_only_label, turn_dialogue_state, last_dialogue_state,
                                    self.config.max_seq_length, self.slot_meta, is_last_turn, self.ontology)

        instances.append(instance)
        last_dialogue_state = turn_dialogue_state
        # print(last_dialogue_state)
    return instances
def BIO_tagging(BIO_INIT, dialogue, value, value_type, slot):
    type_list = ["0", "time", "location", "type", "num", "name", "adj", "bool", "day", "area", "location"]

    for i, word in dialogue:


    Init = 0
    value=list(value.split())
    dialogue=list(dialogue.split())
    turn_dialogue=enumerate(dialogue)
    for idx,token in turn_dialogue:
        if token==value[0]:
           for value_idx,value_token in enumerate(value):
              if value_idx==0:
                 BIO_INIT[value_idx+idx]="B"+"-"+value_type
              else:
                 try:
                   if dialogue[value_idx+idx]==value[value_idx]:
                     BIO_INIT[value_idx+idx]="I"+"-"+value_type
                 except:
                    break
           for skip in range(len(value)):
               try:
                   next(turn_dialogue)
               except:
                    break
    # print('BIO_INIT',BIO_INIT)
    return BIO_INIT

