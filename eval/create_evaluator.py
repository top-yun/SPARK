import os
import json
import numpy as np
import pandas as pd
from config import *
from collections import defaultdict
from eval.utils import *

class BaseEvaluator:
    def __init__(self, root):
        super(BaseEvaluator, self).__init__()

        # Create evaluation results folder
        self.save_dir = os.path.join(root, "eval_results")
        os.makedirs(self.save_dir, exist_ok=True)

    def reset(self):
        # Reset results for new dataset evaluation
        self.gen_answers = []
        self.inputs = []
    
    def process(self, inputs, outputs):
        # Merge results
        self.inputs.extend(inputs)
        self.gen_answers.extend(outputs)

class Evaluator(BaseEvaluator):
    def __init__(self, root):
        """
        Eval Datasets

        - RGB
        - thermal
        - depth
        - CT
        
        """
        super().__init__(root)
    
    def evaluate(self, model, accel):

        # gathering all gpu to one device
        self.inputs = accel.gather_for_metrics(self.inputs)
        self.gen_answers = accel.gather_for_metrics(self.gen_answers)
        
        if accel.is_main_process:
            # Select evaluation for dataset
            return self.evaluate_ALL(model, accel)
        else:
            return None
        
    def evaluate_ALL(self, model, accel):
        sensor_type_groups = defaultdict(list)
        
        for inputs, answer in zip(self.inputs, self.gen_answers):
            entry = {
                'question_id': inputs['id'],
                'sensor_type': inputs['sensor_type'],
                'question_type': inputs['question_type'],
                'question': inputs['question'],
                'question_query': inputs['question_query'],
                'answer': inputs['answer'],
                'prediction': answer
            }
            sensor_type_groups[inputs['sensor_type']].append(entry)
        
        rgb_text = self.evaluate_RGB(model, sensor_type_groups['RGB'], accel)
        thermal_text = self.evaluate_thermal(model, sensor_type_groups['thermal'], accel)
        depth_text = self.evaluate_depth(model,sensor_type_groups['depth'], accel)
        xr_text = self.evaluate_XR(model, sensor_type_groups['xray'], accel)        
        
    
    def evaluate_RGB(self, model, pred_answers , accel):
        # Save results
        pred_pth = os.path.join(self.save_dir, f"{model}_RGB_results.csv")
        accel.print(f"Start evaluating RGB.")
        # Evaluation
        text = evaluation(pred_answers, pred_pth)
        # Save results
        accel.print(f"Finished evaluating RGB. Evaluate the result file saved to {pred_pth}.")
        return text

    def evaluate_thermal(self, model,pred_answers , accel):
       # Save results
        pred_pth = os.path.join(self.save_dir, f"{model}_thermal_results.csv")
        accel.print(f"Start evaluating thermal.")
        # Evaluation
        text = evaluation(pred_answers, pred_pth)
        # Save results
        accel.print(f"Finished evaluating thermal. Evaluate the result file saved to {pred_pth}.")
        return text
    
    def evaluate_depth(self, model,pred_answers , accel):
        # Save results
        pred_pth = os.path.join(self.save_dir, f"{model}_depth_results.csv")
        accel.print(f"Start evaluating depth.")
        # Evaluation
        text = evaluation(pred_answers, pred_pth)
        # Save results
        accel.print(f"Finished evaluating depth. Evaluate the result file saved to {pred_pth}.")
        return text

    def evaluate_XR(self, model, pred_answers ,accel):
        # Save results
        pred_pth = os.path.join(self.save_dir, f"{model}_XR_results.csv")
        accel.print(f"Start evaluating X-ray.")
        # Evaluation
        text = evaluation(pred_answers, pred_pth)
        # Save results
        accel.print(f"Finished evaluating X-ray. Evaluate the result file saved to {pred_pth}.")
        return text