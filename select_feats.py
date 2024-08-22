import json


acc = json.load(open("./history/exp_light_gbm_5_turns_feature_acc.json", "r"))
sorted_acc = dict(sorted(acc.items(), key=lambda item: item[1], reverse=True))

with open("feat_priority.json", "w") as f:
    json.dump(list(sorted_acc.keys()), f)