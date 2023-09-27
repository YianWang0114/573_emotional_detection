import pdb
import argparse
from sklearn.metrics import confusion_matrix
'''
sample command: python3 confusion.py --method bert --set dev --task emotion
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Create confusion matrix for outpus.')
    parser.add_argument("--set", type=str, default='dev')
    parser.add_argument("--method", type=str, default='bert')
    parser.add_argument("--task", type=str, default='emotion')
    args = parser.parse_args()
    return args

def read_label(path, result):
    f = open(path + result, "r")
    data = f.readlines()
    label = []
    for instance in data:
        try:
            tmp = instance.split(',')[1].replace('\n', '')
            label.append(tmp)
        except:
            pass
    return label

def compute_confusion(label, label_true):
    assert (len(label) == len(label_true))
    result = confusion_matrix(label_true, label)
        # labels=["joyful",
        #   "mad",
        #   "neutral",
        #   "peaceful",
        #   "powerful",
        #   "sad",
        #   "scared"]
    return result

def write_emotion(confusion, set, method):
    label_list =  ["joyful  ",
          "mad     ",
          "neutral ",
          "peaceful",
          "powerful",
          "sad     ",
          "scared  "]
    
    with open("../../../outputs/D4/confusion_" + method + "_" + set + ".txt", "w") as f:
        f.writelines('\t\t\t')
        for i in label_list:
            f.writelines(i+'\t')
        f.writelines('\n')
        #f.writelines("\tjoyful  " + "\t" + "mad" + "\t" + "neutral" + "\t" + "peaceful" + "\t" + "powerful" + "\t" + "sad" + "\t" + "scared\n")
        for i in range(len(confusion)):
            f.writelines(label_list[i] + "\t\t\t")
            for j in confusion[i]:
                f.writelines(str(j)+'\t\t\t')
            f.writelines('\n')

def main():
    args = parse_args()
    path = '../../../outputs/D4/'
    if (args.task == 'emotion'):
        result = 'emory_emotion_recognition_' + args.method + '_' + args.set + '.txt'
        true = 'emory_emotion_recognition_true' + '_' + args.set + '.txt'
    else:
        result = 'personality_detection_' + args.method + '_' + args.set + '.txt'
        true = 'personality_detection_true' + '_' + args.set + '.txt'
    label = read_label(path, result)
    true_label = read_label(path, true)
    confusion = compute_confusion(label, true_label)
    if (args.task == 'emotion'):
        write_emotion(confusion, args.set, args.method)



if __name__ == "__main__":
    main()