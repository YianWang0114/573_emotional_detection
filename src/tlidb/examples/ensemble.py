import argparse
import pdb
from sklearn.metrics import f1_score, accuracy_score
'''
sample command: python3 ensemble.py --primary bert --one bow_svm --two lexicon --set dev --task emotion
'''
def parse_args():
    parser = argparse.ArgumentParser(description='This is ensemble method.')
    parser.add_argument("--primary", type=str, default=None)
    parser.add_argument("--one", type=str, default=None)
    parser.add_argument("--two", type=str, default=None)
    parser.add_argument("--three", type=str, default=None)
    parser.add_argument("--set", type=str, default='dev')
    parser.add_argument("--true", type=str, default='true')
    parser.add_argument("--task", type=str, default='emotion')
    args = parser.parse_args()
    return args

def read_label(path, result):
    f = open(path + result, "r")
    data = f.readlines()
    label = []
    id = []
    for instance in data:
        id_emp = instance.split(',')[0]
        tmp = instance.split(',')[1].replace('\n', '')
        label.append(tmp)
        id.append(id_emp)
    return label, id

def ensemble(pri,one,two, three):
    if (three == None):
        assert (len(pri) == len(one) and len(pri) == len(two))
        ens = []
        for i in range(len(pri)):
            if (one[i] == two[i]):
                tmp = one[i]
            else:
                tmp = pri[i]
            ens.append(tmp)
        return ens
    else:
        assert (len(pri) == len(one) and len(pri) == len(two) and len(pri) == len(three))
        ens = []
        for i in range(len(pri)):
            if (one[i] == two[i] == three[i]):
                tmp = one[i]
            elif (one[i] == two[i] and pri[i] != three[i]):
                tmp = one[i]
            elif (one[i] == three[i] and pri[i] != two[i]):
                tmp = one[i]
            elif (two[i] == three[i] and pri[i] != one[i]):
                tmp = two[i]
            else:
                tmp = pri[i]
            ens.append(tmp)
        return ens

def main():
    args = parse_args()
    task = 'emory_emotion_recognition'
    if (args.task == 'personality'):
        task = 'personality_detection'
    primary_result = task + '_' + args.primary + '_' + args.set + '.txt'
    one_result = task + '_' + args.one + '_' + args.set + '.txt'
    two_result = task + '_' + args.two + '_' + args.set + '.txt' 
    true_result = task + '_' + args.true + '_' + args.set + '.txt' 
    path = '../../../outputs/D4/'
    primary_label, id = read_label(path,primary_result)
    one_label, _ = read_label(path,one_result)
    two_label, _ = read_label(path,two_result)
    three_label = None
    if (args.three != None):
        three_result = task + '_' + args.three + '_' + args.set + '.txt' 
        three_label, _ = read_label(path,three_result)
    true_label, _ = read_label(path,true_result)
    ens_label = ensemble(primary_label, one_label, two_label, three_label)
    assert(len(id) == len(ens_label))
    micro_f1 = f1_score(true_label, ens_label, average='micro')
    weighted_f1 = f1_score(true_label, ens_label, average='weighted')
    print('F1_micro: '+str(micro_f1)+'\nF1_weighted: '+str(weighted_f1))
    if (args.three == None):
        with open("../../../outputs/D4/"+task+"_"+args.primary+"_"+args.one+"_"+args.two+"_"+args.set+".txt", "w") as f:
            for i in range(len(id)):
                f.writelines(id[i]+',')
                f.writelines(ens_label[i]+'\n')
            f.writelines("micro_f1: "+str(micro_f1)+'\n')
            f.writelines("weighted_f1: "+str(weighted_f1)+'\n')
    else:
        with open("../../../outputs/D4/"+task+"_4_"+args.set+".txt", "w") as f:
            for i in range(len(id)):
                f.writelines(id[i]+',')
                f.writelines(ens_label[i]+'\n')
            f.writelines("micro_f1: "+str(micro_f1)+'\n')
            f.writelines("weighted_f1: "+str(weighted_f1)+'\n')


if __name__ == "__main__":
    main()





