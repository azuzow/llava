from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import json 
import cv2
import os
import torch

import re

def calculate_accuracy(gt, predictions):
    # Count the number of correct predictions
    correct_predictions = sum([1 for true, pred in zip(gt, predictions) if true == pred])
    # Calculate the accuracy
    accuracy = correct_predictions / len(gt) if gt else 0  # Prevent division by zero
    return accuracy
def find_first_number(text):
    # Use a regular expression to search for the first occurrence of one or more digits
    match = re.search(r'\d+', text)
    if match:
        # If a match is found, return it as an integer
        return int(match.group(0))
    else:
        # If no match is found, return None or an appropriate value indicating no match
        return 0


model_path = "liuhaotian/llava-v1.5-7b"
# torch.cuda.empty_cache()
print('-----------------------------------------------------------')

with open('/home/zuzow/Downloads/init_150questions.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

video_base_path = '/home/zuzow/egoschema-public/videos/videos/'  # Base path where videos are stored
save_frames_path = '/home/zuzow/egoschema-public/frames/'  # Directory to save frameslo



dataset_questions_path = '/home/zuzow/egoschema-public/questions.json'
dataset_answers_path = '/home/zuzow/egoschema-public/subset_answers.json'


# Load the questions
with open(dataset_questions_path, 'r') as file:
    questions = json.load(file)

# Load the answers
with open(dataset_answers_path, 'r') as file:
    answers = json.load(file)
# print(answers)
# Prune questions to only those with answers
pruned_questions_with_answers = [
    {**question, "answer": answers[question['q_uid']]}
    for question in questions if question['q_uid'] in answers
]

# Path for the new subset questions and answers JSON file
subset_questions_answers_path = '/home/zuzow/egoschema-public/subset_questions_answers.json'

# Save the pruned questions with their answers to a file
with open(subset_questions_answers_path, 'w') as file:
    json.dump(pruned_questions_with_answers, file, indent=4)

if not os.path.exists(save_frames_path):
    os.makedirs(save_frames_path)



gt=[]
predictions=[]
counter=0
for item in data:
    q_uid = item['q_uid']
    video_path = os.path.join(video_base_path, f"{q_uid}.mp4")

    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {q_uid}.mp4")
        continue

    # Calculate the halfway frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    halfway_frame = total_frames // 2
    
    # Set the video frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, halfway_frame)
    
    # Read the frame
    success, frame = cap.read()
    if success:
        # Save the frame
        frame_save_path = os.path.join(save_frames_path, f"{q_uid}.jpg")
        cv2.imwrite(frame_save_path, frame)
        print(f"Frame saved to {frame_save_path}")
    else:
        print(f"Failed to extract frame for {q_uid}.mp4")
    
    # Release the video capture object
    cap.release()
    prompt = f"The 180 seconds video is captured by the cameraman, resulting in an egocentric perspective. 'C' stands for the cameraman.I request your selection of the most appropriate response to the following question about the provided image from the five options provided\n\nQuestion: {item['question']} \noption 0: {item['option 0']}\noption 1: {item['option 1']}\noption 2: {item['option 2']}\noption 3: {item['option 3']}\noption 4: {item['option 4']}\n\nyour response must be strictly in the following specified format.\n\n#Selection: write your selection number here.\n#Reasons: state the reasons for your selection in approximately 50 words."
    # prompt = item['question']
    # prompt = 'describe the image'
    # print(len(prompt))
    # print(prompt)
    args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": frame_save_path,
    "sep": ",",
    "temperature": 0.2,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 1024})()
    print('------------------------------------------')
    result = eval_model(args)

    value= find_first_number(result)
    predictions.append(value)
    gt.append(item['CA'])
    print(value,item['CA'])
    print('------------------------------------------')
    counter+=1

    if counter%150==0:
        break

accuracy = calculate_accuracy(gt, predictions)
print(f"Accuracy: {accuracy:.2f}")