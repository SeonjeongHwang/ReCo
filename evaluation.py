from collections import Counter
import evaluate
f1_metric = evaluate.load("f1")

def tl_evaluate(es_types, predictions, references):
    map = {"Word Matching": 0,
           "Paraphrasing": 1,
           "Inference": 2,
           "Transformed Word Matching": 3,
           "Transformed Paraphrasing": 4}

    predictions_5, references_5 = [], []
    predictions_3, references_3 = [], []  
    for es_type, pred, gold in zip(es_types, predictions, references):
        assert pred in map.keys() and gold in map.keys()

        pred = map[pred]
        gold = map[gold]
        
        if es_type == "single":
            predictions_5.append(pred)
            references_5.append(gold)                        
        
        if gold == 0 and pred in [0, 3]:
            pred = 0
        elif gold == 1 and pred in [1, 4]:
            pred = 1
                
        predictions_3.append(pred)
        references_3.append(gold)
            
    score_5 = f1_metric.compute(predictions=predictions_5, references=references_5, labels=[0,1,2,3,4], average='micro')["f1"]
    score_3 = f1_metric.compute(predictions=predictions_3, references=references_3, labels=[0,1,2], average='micro')["f1"]
        
    print(f"F1 Score - 5-level:", round(score_5*100, 1))
    print(f"F1 Score - 3-level:", round(score_3*100, 1))
    scores = {"5-level": round(score_5*100, 1),
              "3-level": round(score_3*100, 1)}
    
    return scores

def tl_inference(predictions, references):
    map = {"Inference": 0,
           "Non-inference": 1}
    
    predictions_final, references_final = [], []    
    for pred, gold in zip(predictions, references):
        assert pred in map.keys()

        pred = map[pred]
        if gold == "Inference":
            gold = 0
        else:
            gold = 1
        
        predictions_final.append(pred)
        references_final.append(gold)
            
    score_final = f1_metric.compute(predictions=predictions_final, references=references_final, labels=[0,1], average='micro')["f1"]
        
    print(f"F1 Score - Inference Detection:", round(score_final*100, 2))
    scores = {"Inference Detection": round(score_final*100, 2)}
    
    return scores

def tl_paraphrasing(predictions, references):
    map = {"Paraphrasing": 0,
           "Word Matching": 1}
    
    predictions_final, references_final = [], []    
    for pred, gold in zip(predictions, references):
        assert pred in map.keys()

        pred = map[pred]
        if "Paraphrasing" in gold:
            gold = 0
        else:
            gold = 1
        
        predictions_final.append(pred)
        references_final.append(gold)
            
    score_final = f1_metric.compute(predictions=predictions_final, references=references_final, labels=[0,1], average='micro')["f1"]
        
    print(f"F1 Score - Paraphrasing Detection:", round(score_final*100, 2))
    scores = {"Paraphrasing Detection": round(score_final*100, 2)}
    
    return scores

def tl_transformation(predictions, references):
    map = {"Transformation": 0,
           "Non-transformation": 1}
    
    predictions_final, references_final = [], []    
    for pred, gold in zip(predictions, references):
        assert pred in map.keys()

        pred = map[pred]
        if "Transformed" in gold:
            gold = 0
        else:
            gold = 1
        
        predictions_final.append(pred)
        references_final.append(gold)
            
    score_final = f1_metric.compute(predictions=predictions_final, references=references_final, labels=[0,1], average='micro')["f1"]
    
    print(f"F1 Score - Transformation Detection:", round(score_final*100, 2))
    scores = {"Transformation Detection": round(score_final*100, 2)}
    
    return scores

def es_evaluate(predictions, references):
    map = {"Insufficient": 0,
           "Single": 1,
           "Inter": 2}

    predictions_final, references_final = [], []    
    for pred, gold in zip(predictions, references):
        assert pred in map.keys() and gold in map.keys()

        pred = map[pred]
        gold = map[gold]
        
        predictions_final.append(pred)
        references_final.append(gold)
            
    score_final = f1_metric.compute(predictions=predictions_final, references=references_final, labels=[0,1,2], average='micro')["f1"]
        
    print(f"F1 Score - Final:", round(score_final*100, 2))
    scores = {"Final": round(score_final*100, 2)}
    
    return scores

def es_falsify(predictions, references):
    map = {"Insufficient": 0,
           "Contradiction": 1}

    predictions_final, references_final = [], []    
    for pred, gold in zip(predictions, references):
        assert pred in map.keys()

        pred = map[pred]
        if gold == "Insufficient":
            gold = 0
        else:
            gold = 1
        
        predictions_final.append(pred)
        references_final.append(gold)
            
    score_final = f1_metric.compute(predictions=predictions_final, references=references_final, labels=[0,1], average='micro')["f1"]
        
    print(f"F1 Score - Falsify:", round(score_final*100, 2))
    scores = {"Falsify": round(score_final*100, 2)}
    
    return scores

def es_cnt_evidence(predictions, references):
    map = {"Insufficient": 0,
           "Single": 1,
           "Inter": 2}

    predictions_final, references_final = [], []    
    for pred, gold in zip(predictions, references):
        assert pred in map.keys() and gold in map.keys()

        pred = map[pred]
        gold = map[gold]
        
        predictions_final.append(pred)
        references_final.append(gold)
            
    score_final = f1_metric.compute(predictions=predictions_final, references=references_final, labels=[0,1,2], average='micro')["f1"]
        
    print(f"F1 Score - Count Evidence:", round(score_final*100, 2))
    scores = {"Count Evidence": round(score_final*100, 2)}
    
    return scores

def qa_evaluate(predictions, references):
    label_map = {"Not True": 0, "True": 1}

    predictions_final, references_final = [], []    
    for pred, gold in zip(predictions, references):
        assert pred in label_map and gold in label_map

        predictions_final.append(label_map[pred])
        references_final.append(label_map[gold])
            
    score_final = f1_metric.compute(
        predictions=predictions_final,
        references=references_final,
        labels=[0, 1],
        average='micro'
    )["f1"]
        
    print(f"F1 Score:", round(score_final * 100, 2))
    scores = {"F1": round(score_final * 100, 2)}
    
    return scores