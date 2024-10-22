import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI
import pandas as pd

def create_demo(white_no_PNA, white_PNA, black_no_PNA, black_PNA):
    ###
    ### Load demo example frame
    ### Choose relevant demo examples
    ### Then create demo prompt and list of demo image paths
    ###
    demo_frame = pd.read_csv('/home/joseph/datasets/chexpertchestxrays-u20210408/chexpert_binaryPNA_demo_df_labels.csv', index_col=0)
    total_samples = white_no_PNA + white_PNA + black_no_PNA + black_PNA
    
    black_frame = demo_frame[demo_frame.binary_race == 'Black']
    black_PNA_frame = black_frame[black_frame.Pneumonia == 1].sample(black_PNA, random_state=42)
    black_no_PNA_frame = black_frame[black_frame.Pneumonia != 1].sample(black_no_PNA, random_state=42)
    
    white_frame = demo_frame[demo_frame.binary_race == 'White']
    white_PNA_frame = white_frame[white_frame.Pneumonia == 1].sample(white_PNA, random_state=42)
    white_no_PNA_frame = white_frame[white_frame.Pneumonia != 1].sample(white_no_PNA, random_state=42)
    
    final_demo_frame = pd.concat([black_PNA_frame,
                                  black_no_PNA_frame,
                                  white_PNA_frame,
                                  white_no_PNA_frame]).sample(total_samples, random_state=42) # sample full num to shuffle
    return final_demo_frame

def main(
    model,
    white_no_PNA, 
    white_PNA, 
    black_no_PNA, 
    black_PNA,
    num_qns_per_round,
    detail="auto",
):
    """
    Run queries for each test case in the test_df dataframe using demonstrating examples sampled from demo_df dataframe.

    model[str]: the specific model checkpoint to use e.g. "Gemini1.5", "gpt-4-turbo-2024-04-09"
    white_no_PNA[int]: number of demonstrating examples to include from White patients w/o PNA
    white_PNA[int]: number of demonstrating examples to include from White patients w/ PNA
    black_no_PNA[int]: number of demonstrating examples to include from Black patients w/o PNA
    black_PNA[int]: number of demonstrating examples to include from Black patients w/ PNA
    num_qns_per_round[int]: number of queries to be batched in one API call
    detail[str]: resolution level for GPT4(V)-series models, not used for Gemini models
    """

#     class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    EXP_NAME = f"chexpertBinaryPNA_{white_no_PNA}_{white_PNA}_{black_no_PNA}_{black_PNA}_{model}_{num_qns_per_round}"
    
    demo_frame = create_demo(white_no_PNA, white_PNA, black_no_PNA, black_PNA)
    test_df = pd.read_csv('/home/joseph/datasets/chexpertchestxrays-u20210408/chexpert_binaryPNA_test_df_labels.csv', index_col=0)

    if model.startswith("gpt"):
        api = GPT4VAPI(model=model, detail=detail)
    else:
        assert model == "Gemini1.5"
        api = GeminiAPI(location=location)
    print(EXP_NAME, f"test size = {len(test_df)}")

    # create demo_examples from my demo_frame
    # list of tuples
    # i[0] = (path_to_image, class name)
    demo_paths = []
    demo_labels = []
    for i,row in demo_frame.iterrows():
        ans_choice = "B" if row.Pneumonia == 1 else "A"
        demo_paths.append(row.updated_path)
        demo_labels.append(ans_choice)
    demo_examples = list(zip(demo_paths, demo_labels))
    
    # Load existing results
    if os.path.isfile(f"{EXP_NAME}.pkl"):
        with open(f"{EXP_NAME}.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    test_df = test_df.sample(frac=1, random_state=66)  # Shuffle the test set
    for start_idx in tqdm(range(0, len(test_df), num_qns_per_round), desc=EXP_NAME):
        end_idx = min(len(test_df), start_idx + num_qns_per_round)

#         random.shuffle(demo_examples)  already shuffled so don't need to do this 
        prompt = "Please respond with the following format for each question, in the form of a single capital letter specifying which label best describes the image. Do not deviate from the format, because it will be automatically parsed."
        image_paths = [
            i[0] for i in demo_examples
        ]
        for demo in demo_examples:
            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question: Which finding best describes the radiograph above?
Choices: A. No Pneumonia, B. Pneumonia
Answer Choice: {demo[1]}
"""
        qns_idx = []
        for idx, i in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
            qns_idx.append(i.Index)
            image_paths.append(i.updated_path)
            qn_idx = idx + 1

            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question {qn_idx}: Which finding best describes the radiograph above?
Choices {qn_idx}: A. No Pneumonia, B. Pneumonia

"""
        for i in range(start_idx, end_idx):
            qn_idx = i - start_idx + 1
            prompt += f"""
Please respond with the following format for each question:
---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---

Do not deviate from the above format. Repeat the format template for the answer."""
        qns_id = str(qns_idx)
        for retry in range(3):
            if (
                (qns_id in results)
                and (not results[qns_id][0].startswith("ERROR"))
                and (
                    f"END FORMAT TEMPLATE FOR QUESTION {end_idx-start_idx}"
                    in results[qns_id][0]
                )
            ):  # Skip if results exist and successful
                continue

            try:
                res = api(
                    prompt,
                    image_paths=image_paths,
                    real_call=True,
                    max_tokens=60 * num_qns_per_round,
                )
            except Exception as e:
                res = f"ERROR!!!! {traceback.format_exc()}"
            except KeyboardInterrupt:
                previous_usage = results.get("token_usage", (0, 0, 0))
                total_usage = tuple(
                    a + b for a, b in zip(previous_usage, api.token_usage)
                )
                results["token_usage"] = total_usage
                with open(f"{EXP_NAME}.pkl", "wb") as f:
                    pickle.dump(results, f)
                exit()

            print(res)
            results[qns_id] = (res,prompt,image_paths)

    # Update token usage and save the results
    previous_usage = results.get("token_usage", (0, 0, 0))
    total_usage = tuple(a + b for a, b in zip(previous_usage, api.token_usage))
    results["token_usage"] = total_usage
    with open(f"{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(results, f)


        
if __name__ == "__main__":
    
    main("gpt-4o-2024-05-13",
    50, 
    50, 
    0, 
    0,
    50,)
