import evaluate
f1_metric = evaluate.load("f1")


def compute_f1(predictions, references, labels, average):
    return f1_metric.compute(
        predictions=predictions,
        references=references,
        labels=labels,
        average=average
    )["f1"]


def make_scores(predictions, references, labels, prefix=""):
    micro = compute_f1(predictions, references, labels, "micro")
    macro = compute_f1(predictions, references, labels, "macro")

    def key(name):
        return f"{prefix}_{name}" if prefix else name

    #print(f"F1 Score{' - ' + prefix if prefix else ''} | micro: {round(micro*100, 2)}, macro: {round(macro*100, 2)}")

    return {
        key("micro_f1"): round(micro * 100, 2),
        key("macro_f1"): round(macro * 100, 2),
    }


def tl_evaluate(es_types, predictions, references):
    label_map = {
        "Word Matching": 0,
        "Paraphrasing": 1,
        "Inference": 2,
        "Transformed Word Matching": 3,
        "Transformed Paraphrasing": 4,
    }

    predictions_5, references_5 = [], []
    predictions_3, references_3 = [], []

    for es_type, pred, gold in zip(es_types, predictions, references):
        assert pred in label_map and gold in label_map

        pred_idx = label_map[pred]
        gold_idx = label_map[gold]

        if es_type == "single":
            predictions_5.append(pred_idx)
            references_5.append(gold_idx)

        # Collapse transformed variants for 3-level scoring
        if gold_idx == 0 and pred_idx in [0, 3]:
            pred_idx = 0
        elif gold_idx == 1 and pred_idx in [1, 4]:
            pred_idx = 1

        predictions_3.append(pred_idx)
        references_3.append(gold_idx)

    scores = {}
    scores.update(make_scores(predictions_5, references_5, labels=[0, 1, 2, 3, 4], prefix="5-level"))
    scores.update(make_scores(predictions_3, references_3, labels=[0, 1, 2],       prefix="3-level"))

    return scores


def tl_inference(predictions, references):
    label_map = {"Inference": 0, "Non-inference": 1}

    preds = [label_map[p] for p in predictions]
    golds = [0 if g == "Inference" else 1 for g in references]

    return make_scores(preds, golds, labels=[0, 1], prefix="inference_detection")


def tl_paraphrasing(predictions, references):
    label_map = {"Paraphrasing": 0, "Word Matching": 1}

    preds = [label_map[p] for p in predictions]
    golds = [0 if "Paraphrasing" in g else 1 for g in references]

    return make_scores(preds, golds, labels=[0, 1], prefix="paraphrasing_detection")


def tl_transformation(predictions, references):
    label_map = {"Transformation": 0, "Non-transformation": 1}

    preds = [label_map[p] for p in predictions]
    golds = [0 if "Transformed" in g else 1 for g in references]

    return make_scores(preds, golds, labels=[0, 1], prefix="transformation_detection")


def es_evaluate(predictions, references):
    label_map = {"Insufficient": 0, "Single": 1, "Inter": 2}

    preds = [label_map[p] for p in predictions]
    golds = [label_map[g] for g in references]

    return make_scores(preds, golds, labels=[0, 1, 2])


def es_falsify(predictions, references):
    label_map = {"Insufficient": 0, "Contradiction": 1}

    preds = [label_map[p] for p in predictions]
    golds = [0 if g == "Insufficient" else 1 for g in references]

    return make_scores(preds, golds, labels=[0, 1], prefix="falsify")


def es_cnt_evidence(predictions, references):
    label_map = {"Insufficient": 0, "Single": 1, "Inter": 2}

    preds = [label_map[p] for p in predictions]
    golds = [label_map[g] for g in references]

    return make_scores(preds, golds, labels=[0, 1, 2], prefix="count_evidence")


def qa_evaluate(predictions, references):
    label_map = {"Not True": 0, "True": 1}

    preds = [label_map[p] for p in predictions]
    golds = [label_map[g] for g in references]

    return make_scores(preds, golds, labels=[0, 1])