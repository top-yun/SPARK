from collections import defaultdict
import pandas as pd
import re

def extract_answer(prediction):
    if len(prediction) == 0 :
        return ''
    
    if prediction[0].lower() in 'abcd':
        return prediction[0].upper()
    
    if len(prediction) >= 2 and prediction[:2].lower() in 'no':
        return 'no'
    
    if len(prediction) >= 3 and prediction[:3].lower() in 'yes':
        return 'yes'
    
    pattern_with_period = r'\b[A-D]\b'
    matches_with_period = re.findall(pattern_with_period, prediction)
    if len(matches_with_period) > 0:
        return matches_with_period[0][0].upper()
    
    return ''

def evaluation(pred_answers, pred_pth):
    accuracy_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})

    for entry in pred_answers:
        question_type = entry['question_type']
        predction = extract_answer(entry['prediction'])
        if entry['answer'] == predction :
            accuracy_by_type[question_type]['correct'] += 1
            entry['correct'] = True
        else :
            entry['correct'] = False
        accuracy_by_type[question_type]['total'] += 1

    accuracy_scores = {q_type: (stats['correct'] / stats['total']) for q_type, stats in accuracy_by_type.items()}
    
    pd.DataFrame(pred_answers).to_csv(pred_pth, index=False)

    text = ''
    for q_type, accuracy in sorted(accuracy_scores.items()):
        text += f"{q_type}: {accuracy:.2%}\n"
        print(f"Question Type: {q_type}, Accuracy: {accuracy:.2%}")
    return text