####################
# references:
# [1] https://github.com/DART-Laboratory/Flash-IDS
# [2] Our contribution
####################

# ref. [1]
def _get_adjacent(ids, mapp, edges, hops):
    if hops == 0:
        return set()
    
    neighbors = set()
    for edge in zip(edges[0], edges[1]):
        if any(mapp[node] in ids for node in edge):
            neighbors.update(mapp[node] for node in edge)

    if hops > 1:
        neighbors = neighbors.union(_get_adjacent(neighbors, mapp, edges, hops - 1))
    
    return neighbors

# ref. [1], [2]
def calculate_metrics(TP, FP, FN, TN):
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0

    prec = TP / (TP + FP) if TP + FP > 0 else 0
    rec = TP / (TP + FN) if TP + FN > 0 else 0
    fscore = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0
    acc = (TN + TP) / (TN + TP + FN + FP) if TN + TP + FN + FP > 0 else 0

    return prec, rec, fscore, FPR, TPR, acc

# ref. [1], [2]
def eval_helper(MP, all_pids, GP, edges, mapp):
    TP = MP.intersection(GP)
    FP = MP - GP
    FN = GP - MP
    TN = all_pids - (GP | MP)

    two_hop_gp = _get_adjacent(GP, mapp, edges, 2)
    two_hop_tp = _get_adjacent(TP, mapp, edges, 2)
    FPL = FP - two_hop_gp
    TPL = TP.union(FN.intersection(two_hop_tp))
    FN = FN - two_hop_tp

    TP, FP, FN, TN = len(TPL), len(FPL), len(FN), len(TN)

    prec, rec, fscore, FPR, TPR, acc = calculate_metrics(TP, FP, FN, TN)
    print(f"True Positives/False Positives/False Negatives/True Negatives: {TP}/{FP}/{FN}/{TN}")
    print(f"Accuracy: {round(acc, 2)}, Precision: {round(prec, 2)}, Recall: {round(rec, 2)}, Fscore: {round(fscore, 2)}")
    
    return TPL, FPL
