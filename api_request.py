import json
import os
import requests
import tqdm
# Replace "your-server-ip" with your server's IP address or use 127.0.0.1 for local execution.
url = "http://0.0.0.0:7889/api/predict/"
headers = {"Content-Type": "application/json"}
def request(video_path, text):
    INSTRUCTION = f"""
    You are a 'Therapist' analyzing a video session. The client said: {text}

    Your tasks:
    1. Briefly describe and understand the context of the conversation.
    2. Predict the following (choose only one for each):
    - [Client's emotion]: anger, sadness, disgust, depression, neutral, joy, fear
    - [Therapist's emotion]: anger, sadness, disgust, depression, neutral, joy, fear
    - [Therapist's strategy]: open question, approval, self-disclosure, restatement, interpretation, advisement, communication skills, structuring the therapy, guiding the pace and depth of the conversation, others
        - Communication skills: Small talk, daily communication, positive atmosphere.
        - Advisement: Guidance or advice for client's issues.
        - Structuring the therapy: Set clear framework, goals, rules.
        - Guiding the pace and depth: Regulate session flow and depth.
        - Others: Support strategies not listed above.
    3. [generate] [Therapist's response] Compose an empathetic response as the therapist. Your reply should:
    - Understand and acknowledge the client's emotion and perspective.
    - Express sympathy for negative situations or approval for positive ones.
    - Avoid negative triggers (disgust, resentment, discrimination, hatred, etc.).
    - Be truthful, supportive, and foster understanding and comfort.
    - Repeat a few words from the client's utterance.
    - Express a different opinion if needed, but never hurt the client's feelings.
    - Safeguard human autonomy, identity, and data dignity.

    Your entire answer must be a single line, return your predictions of 4 tasks including Client's emotion, Therapist's emotion, Therapist's strategy, Therapist's response on 1 line only and separated them by `;`
    Do NOT include any extra words, labels, explanations, or line breaks, only include your predictions.
    """
    data = {
        "data": [
            video_path,
            INSTRUCTION
        ]
    }
    

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()["data"][0]

def main():
    videos_path = "data/video_data"
    jsonl_path = "data/test.jsonl"

    # read jsonl into list of dict
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    preds = []
    # i = 0
    for item in tqdm.tqdm(data):
        text = " ".join(item["utt_user_most_recent"])
        video = item["path_to_vid_user_most_recent"][-1]

        res = request(os.path.join(videos_path, video), text)
        print(res)

        preds.append(res)


    output_pred = "pred.csv"

    with open(output_pred, "w") as f:
        for pred in preds:
            f.write(pred + "\n\n")

main()