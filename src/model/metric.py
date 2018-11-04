

def MRR(right_answer_scores, wrong_answer_scores):
    all_scores_ranked = sorted(right_answer_scores + wrong_answer_scores, reverse=True)
    mrr = 0
    for right_score in right_answer_scores:
        rank = all_scores_ranked.index(right_score) + 1
        mrr += 1.0 / rank
    return mrr / len(right_answer_scores) if len(right_answer_scores) else 0.0


def MAP(right_answer_scores, wrong_answer_scores):
    all_scores_ranked = sorted(right_answer_scores + wrong_answer_scores, reverse=True)
    right_answer_scores_ranked = sorted(right_answer_scores.copy(), reverse=True)
    map = 0
    for idx, right_score in enumerate(right_answer_scores_ranked):
        rank = all_scores_ranked.index(right_score) + 1
        map += (idx + 1) / rank
    return map / len(right_answer_scores) if len(right_answer_scores) else 0.0
