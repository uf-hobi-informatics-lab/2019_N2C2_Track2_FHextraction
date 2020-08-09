import os
import re
from utils import pkl_save, pkl_load, load_text, save_text
from collections import defaultdict
from pathlib import Path
from collections import Counter
from itertools import permutations
from config import (NER_TYPING_ROOT, TEXT_AS_SENT_DIR, PREPROCESSED_TEXT_DIR,
                    OBN_TEST, LSS_TEST, FMS_TEST, CLS_OUTPUT_ROOT, REL_TESTa, REL_OUTPUT_ROOTa,
                    REL_TEST, REL_OUTPUT_ROOT, ENSEMBLE_THRESHOLD, GLOBAL_CUTOFF,
                    PRED_SUBTASK_1, PRED_SUBTASK_2)


SCORE = {'Yes': 2, 'NA': 1, 'No': 0}
en_map = {'FAMILYMEMBER': 'FamilyMember', 'OBSERVATION': 'Observation',
          'familymember': 'FamilyMember', 'observation': 'Observation',
          'FamilyMember': 'FamilyMember', 'Observation': 'Observation',
          'LivingStatus': 'LivingStatus', 'LIVINGSTATUS': 'LivingStatus',
          'livingstatus': 'LivingStatus'}


def living_status_tag2score(tag1, tag2):
    s1 = SCORE[tag1]
    s2 = SCORE[tag2]
    return f"{s1*s2}"


def to_tsv(data, ofn):
    header = "\t".join([str(i+1) for i in range(len(data[0]))])
    with open(ofn, "w") as f:
        f.write(f"{header}\n")
        for each in data:
            d = "\t".join(each)
            f.write(f"{d}\n")


def creat_sent_altered_boundary(sents):
    sb = dict()
    for idx, sent in enumerate(sents):
        s = sent[0][-1][0]
        e = sent[-1][-1][-1]
        sb[idx] = (s, e)
    return sb


def insert_token_and_creat_text_for_testing(sent, token_span):
    new_sents = []
    s, e = token_span

    new_tokens = []

    for token in sent:
        token_text = token[0]
        ts = token[-1][0]
        te = token[-1][-1]

        if ts == s:
            new_tokens.append("[ES]")

        new_tokens.append(token_text)

        if te == e:
            new_tokens.append("[EE]")

    return " ".join(new_tokens)


def text2sents(text, original_offset_only=False):
    nsents = []
    for line in text.split("\n\n"):
        nsent = []
        for word in line.split("\n"):
            word = word.split(" ")
            if original_offset_only:
                word = word[:3] + [word[-1]]
            else:
                word = word[:5] + [word[-1]]
            nsent.append(word)
        nsents.append(nsent)
    return nsents


def non_integrated_results(root, test_fids, original_offset_only=False):
    p = Path(root)
    ll = len(test_fids)
    result = dict()
    for fid in p.glob("*.txt"):
        fid_stem = fid.stem.split(".")[0]
        assert fid_stem in test_fids, f"{fid.stem} is not a test fid"
        ll -= 1
        cont = load_text(fid)
        sents = text2sents(cont.strip(), original_offset_only)
        result[fid_stem] = sents
    assert ll == 0, f"missing {ll} prediction files"
    return result


def extract_entities(res, test_fids, original_offset_only=False):
    entities = defaultdict(list)

    for tfid in test_fids:
        sents = res[tfid]
        term, start, end, start_a, end_a, sematics = [], 0, 0, 0, 0, ""
        prev_tag = "O"

        for sent in sents:
            for word in sent:
                text = word[0]
                orign_pos = (int(word[1]), int(word[2]))

                if original_offset_only:
                    altered_pos = (int(word[1]), int(word[2]))
                else:
                    altered_pos = (int(word[3]), int(word[4]))

                cur_tag = word[-1]
                bounding_sematic = cur_tag.split("-")
                en_so = orign_pos[0]
                en_eo = orign_pos[1]
                en_sa = altered_pos[0]
                en_ea = altered_pos[1]

                if bounding_sematic[0] == "O":
                    # tag is O
                    if prev_tag[-1] != "O":
                        entities[tfid].append((" ".join(term), sematics, (start, end), (start_a, end_a)))
                        term, start, end, start_a, end_a, sematics = [], 0, 0, 0, 0, ""
                else:
                    bounding = bounding_sematic[0]
                    sematic = bounding_sematic[1]
                    if bounding == "B":
                        if prev_tag[-1] != "O":
                            entities[tfid].append((" ".join(term), sematics, (start, end), (start_a, end_a)))
                            term, start, end, start_a, end_a, sematics = [], 0, 0, 0, 0, ""
                        term.append(text)
                        start = en_so
                        end = en_eo
                        start_a = en_sa
                        end_a = en_ea
                        sematics = sematic
                    elif bounding == "I":
                        if prev_tag[-1] == "O":
                            start_a = en_sa
                            sematics = sematic
                        end = en_eo
                        end_a = en_ea
                        term.append(text)
                prev_tag = cur_tag

            if len(term) > 0:
                entities[tfid].append((" ".join(term), sematics, (start, end), (start_a, end_a)))
    return entities


def valida_by_sent(s):
    if "[ES]" not in s or "[EE]" not in s:
        return True
    elif len(re.findall("\[[A-Z]{2}\]", s)) != 2:
        return True
    return False


def get_sent_idx(en_span, sent_bound):
    # (a, b) {1: (c, d)}
    a, b = en_span
    for k, v in sent_bound.items():
        c, d = v
        if a >= c and b <= d:
            return k
        elif (a + 1) >= c and b <= d:
            return k
    return None


def load_ner_typing_results(fdir):
    file_name = "test_results.txt"
    with open(os.path.join(fdir, file_name), "r") as f:
        res = f.read().strip()
    return res.split("\n")


def merge_test_with_result(test_data, *result_data):
    """
        input is the ner typing test input data merged with output
        e.g., 'doc_166@0', "The couple ' s five - year - old [ES] son [EE] has cancer ."], 'NA', 'Son'
        using the first element in test data input to obtain the doc # and entitiy #: doc_166@0 => doc_166 and en_#:0
    """
    d = defaultdict(list)
    num = len(result_data)
    for i, each in enumerate(result_data):
        assert len(test_data) == len(each), f"test input is not same size as result output: {len(test_data)} {len(each)}"
    for each in zip(test_data, *result_data):
        doc_id, en_id = each[0][0].split("@")
        en_id = int(en_id)
        tags = []
        for idx in range(num):
            tags.append(each[idx+1])
        d[doc_id].append((en_id, tags))
    return d


def bio2typing(res_dir, test_fids, tag=0):
    res = non_integrated_results(res_dir, test_fids)
    merged_entities = extract_entities(res, test_fids)
    ner_typing_root = Path(NER_TYPING_ROOT)
    ner_typing_root.mkdir(parents=True, exist_ok=True)
    pkl_save(merged_entities, ner_typing_root / f"merged_entities_{tag}.pkl")
    for test_fid in test_fids:
        pre_txt = load_text(Path(PREPROCESSED_TEXT_DIR) / f"{test_fid}.preprocessed.txt").split("\n")
        sents = pkl_load(Path(TEXT_AS_SENT_DIR) / f"{test_fid}.sents.pkl")
        sent_bound = creat_sent_altered_boundary(sents)

        ens = merged_entities[test_fid]
        fm, ls, ob = [], [], []
        for en_idx, en in enumerate(ens):
            # ('son', 'FAMILYMEMBER', (334, 337), (342, 345))
            en_span = en[-1]
            en_type = en[1].lower()
            sidx = get_sent_idx(en_span, sent_bound)
            en_loc_sent = sents[sidx]
            pure_text = pre_txt[sidx]
            tagged_sent = insert_token_and_creat_text_for_testing(en_loc_sent, en_span)

            if valida_by_sent(tagged_sent):
                print(test_fid, en, tagged_sent)

            if en_type == "familymember":
                fm.append([f"{test_fid}@{en_idx}", f"{test_fid}@{en_idx}", pure_text, tagged_sent])
            elif en_type == "observation":
                ob.append([f"{test_fid}@{en_idx}", pure_text, tagged_sent])
            elif en_type == "livingstatus":
                ls.append([f"{test_fid}@{en_idx}", pure_text, tagged_sent])
            else:
                raise RuntimeError(f"{en_type} is not recognized for {en}")

        # # fms, fmr share the same dir
        pfm = Path(FMS_TEST.format(tag))
        pfm.mkdir(exist_ok=True, parents=True)
        to_tsv(fm, pfm / "test.tsv")

        pfo = Path(OBN_TEST.format(tag))
        pfo.mkdir(exist_ok=True, parents=True)
        to_tsv(ob, pfo / "test.tsv")

        pfl = Path(LSS_TEST.format(tag))
        pfl.mkdir(exist_ok=True, parents=True)
        to_tsv(ls, pfl / "test.tsv")

        pkl_save(fm, ner_typing_root / f"fm_{tag}.pkl")
        pkl_save(ob, ner_typing_root / f"ob_{tag}.pkl")
        pkl_save(ls, ner_typing_root / f"ls_{tag}.pkl")


def gen_res_for_subtask1(test_fids):
    typed_entities_en = []

    for tag in range(5):
        ner_typing_root = Path(NER_TYPING_ROOT)

        merged_entities = pkl_load(ner_typing_root / f"merged_entities_{tag}.pkl")
        fm_test_input = pkl_load(ner_typing_root / f"fm_{tag}.pkl")
        ob_test_input = pkl_load(ner_typing_root / f"ob_{tag}.pkl")
        ls_test_input = pkl_load(ner_typing_root / f"ls_{tag}.pkl")

        fms_test = load_ner_typing_results(CLS_OUTPUT_ROOT.format("fms", tag))
        fmr_test = load_ner_typing_results(CLS_OUTPUT_ROOT.format("fmr", tag))
        obn_test = load_ner_typing_results(CLS_OUTPUT_ROOT.format("obn", tag))
        lss_test = load_ner_typing_results(CLS_OUTPUT_ROOT.format("lss", tag))

        fm_merged_res = merge_test_with_result(fm_test_input, fms_test, fmr_test)
        ob_merged_res = merge_test_with_result(ob_test_input, obn_test)
        ls_merged_res = merge_test_with_result(ls_test_input, lss_test)

        typed_entities = defaultdict(list)
        for tfid in test_fids:
            entities = merged_entities[tfid]
            mrs = fm_merged_res[tfid] + ob_merged_res[tfid] + ls_merged_res[tfid]
            for each in mrs:
                en_id, ner_types = each
                en = entities[en_id]
                typed_entities[tfid].append((en, ner_types, en[1]))

        nd = []
        for k, v in typed_entities.items():
            # (('sister', 'FamilyMember', (576, 582), (597, 603)), ['NA', 'Sister'], 'FamilyMember')
            for each in v:
                nd.append((k, each[0], tuple(each[1]), each[2]))
        typed_entities_en.extend(nd)

    final_res = [e[0] for e in Counter(typed_entities_en).most_common() if e[1] > ENSEMBLE_THRESHOLD]

    typed_entities_f = defaultdict(list)
    for each in final_res:
        typed_entities_f[each[0]].append((each[1], list(each[2]), each[3]))

    task1_ens = []
    for doc_id, ens in typed_entities_f.items():
        for en in ens:
            new_en = [doc_id]
            en_type = en[-1]
            if en_type.upper() == 'FAMILYMEMBER':
                new_en.append(en_map[en_type])
                new_en.append(en[1][1])
                new_en.append(en[1][0])
                task1_ens.append(new_en)
            elif en_type.upper() == "OBSERVATION":
                new_en.append(en_map[en_type])
                new_en.append(en[0][0])
                task1_ens.append(new_en)

    pkl_save(typed_entities_f, Path(NER_TYPING_ROOT) / "typed_entities_ens.pkl")

    task1_ens = sorted({tuple(e) for e in task1_ens}, key=lambda x: int(x[0].split("_")[-1]))

    with open(PRED_SUBTASK_1, "w") as f:
        for each in task1_ens:
            cont = "\t".join(each)
            f.write(f"{cont}\n")


def insert_helper(sent, token_span, ll):
    new_sents = {"s": -1, "e": -1}

    s, e = token_span
    prev = "O"
    ee_counter = True
    es_counter = True

    for idx, each in enumerate(sent):
        t = each[-1][0]
        ts, te = each[1][0], each[1][1]

        if ts >= s and te <= e and t == "B":
            new_sents["s"] = idx
            es_counter = False
        elif ts >= s and te <= e + 1 and t == "B":
            new_sents["s"] = idx
            es_counter = False
        elif ts >= s - 1 and te <= e - 1 and t == "B":
            new_sents["s"] = idx
            es_counter = False
        elif ts >= e and ee_counter:
            new_sents["e"] = idx
            ee_counter = False
        prev = t

    if es_counter:
        ee_index = new_sents["e"]
        es_index = ee_index - ll
        new_sents["s"] = es_index

    if ee_counter:
        new_sents["e"] = len(sent) - 1  # # used to be idx
    return new_sents


def to_relation_text(sent, ids, tag1, tag2):
    """
    training plan 1 bert : sentence with en1 is the sent1 and sentence with en2 is the sent2 for the bert model (sent 1 and 2 may be same sentence with different special tag)
    training plan 2 bert : only use one sentence with all special tags in it. concat sentence if entities are not in the same sentence
    """
    new_sent = []
    pure_sent = []
    s, e = ids['s'], ids['e']
    if s < 0:
        s = e
        e = e + 1

    ll = len(sent)

    for idx, word in enumerate(sent):
        word_text = word[0]
        if idx == s:
            new_sent.append(tag1)
            new_sent.append(word_text)
        elif idx == e:
            new_sent.append(tag2)
            new_sent.append(word_text)
        else:
            new_sent.append(word_text)
        pure_sent.append(word_text)

    if e == ll:
        new_sent.append(tag2)

    return " ".join(new_sent), " ".join(pure_sent)


def valid_relations(s, tag1, tag2):
    if tag1 not in s or tag2 not in s:
        print(s)
        return True
    elif len(re.findall("\[[A-Z]{2}\]", s)) != 2:
        print(s)
        return True
    return False


def insert_tags_for_relation(sents_en1, sents_en2, en1, en2):
    """
        use [ES] before entity1use [EE] after entity1
        use [SS] before entity1use [SE] after entity2
    """
    en1t2, en1t1, en2t2, en2t1 = "[EE]", "[ES]", "[SE]", "[SS]"

    en1span = en1[2]
    en2span = en2[2]
    en1_ll = len(en1[0].split(" "))
    en2_ll = len(en2[0].split(" "))

    en1idx = insert_helper(sents_en1, en1span, en1_ll)
    en2idx = insert_helper(sents_en2, en2span, en2_ll)

    s1, ps1 = to_relation_text(sents_en1, en1idx, en1t1, en1t2)
    if valid_relations(s1, en1t1, en1t2):
        print("Warning with relation: ", sents_en1, en1, en1idx)

    s2, ps2 = to_relation_text(sents_en2, en2idx, en2t1, en2t2)
    if valid_relations(s2, en2t1, en2t2):
        print("Warning with relation: ", sents_en2, en2, en2idx)

    return s1, s2, ps1, ps2


def range_idx(s1, s2):
    sp_tags = {"[EE]", "[ES]", "[SE]", "[SS]"}
    idx1 = []
    idx2 = []
    for idx, w in enumerate(s1):
        if w in sp_tags:
            idx1.append([idx, w])
    for idx, w in enumerate(s2):
        if w in sp_tags:
            idx2.append([idx, w])
    i11, i12 = idx1
    i21, i22 = idx2

    if i21[0] + 1 >= i12[0]:
        i21[0] += 2
        i22[0] += 2
        return [i11, i12, i21, i22]
    elif i11[0] + 1 >= i22[0]:
        i11[0] += 2
        i12[0] += 2
        return [i21, i22, i11, i12]
    else:
        if i21[0] < i11[0] < i12[0] < i22[0]:
            i11[0] += 1
            i12[0] += 1
            return [i21, i22, i11, i12]
        elif i11[0] < i21[0] < i22[0] < i12[0]:
            i21[0] += 1
            i22[0] += 1
            return [i11, i12, i21, i22]


def generate_bert_relation_without_extra_sentence(sents_en1, sents_en2, en1, en2, sents, idx1, idx2):
    en1t2, en1t1, en2t2, en2t1 = "[EE]", "[ES]", "[SE]", "[SS]"

    diff = abs(idx1 - idx2)

    en1span = en1[2]
    en2span = en2[2]
    en1_ll = len(en1[0].split(" "))
    en2_ll = len(en2[0].split(" "))

    en1idx = insert_helper(sents_en1, en1span, en1_ll)
    en2idx = insert_helper(sents_en2, en2span, en2_ll)

    s1, ps1 = to_relation_text(sents_en1, en1idx, en1t1, en1t2)
    if valid_relations(s1, en1t1, en1t2):
        print(sents_en1, en1, en1idx)

    s2, ps2 = to_relation_text(sents_en2, en2idx, en2t1, en2t2)
    if valid_relations(s2, en2t1, en2t2):
        print(sents_en2, en2, en2idx)

    if diff == 1:
        return " ".join([s1, s2])
    elif diff == 0:
        s = [w[0] for w in sents[idx1]]
        s1 = s1.split(" ")
        s2 = s2.split(" ")
        assert len(s1) == len(s2), f"in generate_bert_relation_without_extra_sentence:  {s}\t{ps1}\t{ps2}"

        if not range_idx(s1, s2):
            print(s1)
            print(s2)
            print(en1)
            print(en2)

        for each in range_idx(s1, s2):
            s.insert(each[0], each[1])
        return " ".join(s)
    else:
        s = ""
        if idx1 < idx2:
            for ii in range(idx1 + 1, idx2):
                s += " ".join([w[0] for w in sents[ii]])
                return " ".join([s1, s, s2])
        else:
            for ii in range(idx2 + 1, idx1):
                s += " ".join([w[0] for w in sents[ii]])
                return " ".join([s2, s, s1])


def bio2relation():
    TAG = "pred"
    sdiff = []
    relation_types = []
    pred_relations_plan1 = []
    pred_relations_plan2 = []
    mapping = []

    typed_entities = pkl_load(Path(NER_TYPING_ROOT) / "typed_entities_ens.pkl")
    for doc_id, ens in typed_entities.items():
        pre_txt = load_text(Path(PREPROCESSED_TEXT_DIR) / f"{doc_id}.preprocessed.txt").split("\n")
        sents = pkl_load(Path(TEXT_AS_SENT_DIR) / f"{doc_id}.sents.pkl")
        sent_bound = creat_sent_altered_boundary(sents)
        enids = range(len(ens))
        all_pairs = []
        for e1, e2 in permutations(enids, 2):
            all_pairs.append((e1, e2))

        for each in all_pairs:
            eid1, eid2 = each
            # (('son', 'FAMILYMEMBER', (334, 337), (342, 345)), ['NA', 'Son'], 'FAMILYMEMBER')
            en1 = ens[eid1]
            en2 = ens[eid2]
            if en1[-1].upper() != "FAMILYMEMBER" or en2[-1].upper() == "FAMILYMEMBER":
                continue

            sie1 = get_sent_idx(en1[0][3], sent_bound)
            sie2 = get_sent_idx(en2[0][3], sent_bound)
            if abs(sie1 - sie2) > GLOBAL_CUTOFF:
                continue

            bert_rels = insert_tags_for_relation(sents[sie1], sents[sie2], en1[0], en2[0])
            tagged_s1, tagged_s2, pure_text1, pure_text2 = bert_rels
            pred_relations_plan1.append(
                [TAG, tagged_s1, tagged_s2, pure_text1, pure_text2, f"{abs(sie1 - sie2)}", str()])
            tp = generate_bert_relation_without_extra_sentence(sents[sie1], sents[sie2],
                                                                en1[0], en2[0],
                                                                sents, sie1,sie2)
            pred_relations_plan2.append([TAG, tp, f"{abs(sie1 - sie2)}"])
            mapping.append((doc_id, en1, en2))

    prel = Path(REL_TEST)
    prel.mkdir(parents=True, exist_ok=True)
    pkl_save(mapping, prel / "relation_mappings.tsv")
    to_tsv(pred_relations_plan2, prel / "test.tsv")

    prel = Path(REL_TESTa)
    prel.mkdir(parents=True, exist_ok=True)
    pkl_save(mapping, prel / "relation_mappings.tsv")
    to_tsv(pred_relations_plan1, prel / "test.tsv")


def to_task2_output(output, ofn):
    s = set()
    with open(ofn, "w") as f:
        for each in output:
            res = "\t".join(each)
            if res not in s:
                s.add(res)
                f.write(res)
                f.write("\n")
            else:
                pass


def format_bert_output(pred, cmap):
    assert len(pred) == len(
        cmap), f"the prediction have different number of results with mappings as pred: {len(pred)}; mappings: {len(cmap)}."
    pos_res = []
    for flag, rel in zip(pred, cmap):
        if flag == "pos":
            o = [rel[0], rel[1][1][1], rel[1][1][0]]
            #             print(rel[2][-1])
            if rel[2][-1] == "OBSERVATION" or rel[2][-1] == "Observation":
                o.extend([en_map[rel[2][-1]], rel[2][0][0], rel[2][1][0]])
            elif rel[2][-1] == "LIVINGSTATUS" or rel[2][-1] == "LivingStatus":
                o.extend([en_map[rel[2][-1]], rel[2][1][0]])
            pos_res.append(o)

    return sorted(pos_res, key=lambda x: x[0])


def load_bert_results(ifn):
    res = []
    with open(ifn, "r") as f:
        for line in f.readlines():
             res.append(line.strip())
    return res


def gen_res_for_subtask2():
    mapping = pkl_load(os.path.join(REL_TEST, "relation_mappings.tsv"))
    rel_preds = load_bert_results(Path(REL_OUTPUT_ROOT) / "test_results.txt")
    rel_res = format_bert_output(rel_preds, mapping)
    to_task2_output(rel_res, PRED_SUBTASK_2)

    # mapping = pkl_load(os.path.join(REL_TESTa, "relation_mappings.tsv"))
    # rel_preds = load_bert_results(Path(REL_OUTPUT_ROOTa) / "test_results.txt")
    # rel_res = format_bert_output(rel_preds, mapping)
    # to_task2_output(rel_res, PRED_SUBTASK_2)
