import json
import os
import requests

# Replace "your-server-ip" with your server's IP address or use 127.0.0.1 for local execution.
url = "http://0.0.0.0:7889/api/predict/"
headers = {"Content-Type": "application/json"}
def request(video_path, text):
    INSTRUCTION = f"""
        In the video, the client said: {text}
        
        Now you are the 'Therapist' and you need to understand the context to predict the Client’s emotion, Therapist’s emotion, then Therapist’s strategy. After that you need to make an empathy response to the 'Client' based on the context. Let's think about it step by step: \n
        Step 1: Describe and understanding the context and content of the conversation \n
        Step 2: Predict the following and explain why for each components, notice that Therapist's strategy have guideline: \n
            [emotion] Predict Client's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). \n
            [emotion] Therapist's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). \n
            [strategy] Therapist's strategy: Choose only one from (open question, approval, self-disclosure, restatement, interpretation, advisement, communication skills, structuring the therapy, guiding the pace and depth of the conversation, others). \n
            Guide for Therapist's strategy: Communication Skills: Involve small talk and daily communication with clients, along with using simple phrases and body language during listening, thereby establishing a positive communication atmosphere.; Advisement: Offering guidance, advice, or possible solutions to clients, helping them address psychological issues and emotional distress they encounter.; Structuring the therapy: The therapist sets a clear framework and structure for the therapy process. This includes defining the goals of therapy, its duration, the main activities during therapy, and rules, etc.; Guiding the Pace and Depth of the Conversation: Therapists regulate the pace and depth of conversations during sessions through various techniques and methods, such as shifting the topic when the client is emotionally unstable or guiding the conversation back to key themes when necessary.; Others: Employ support strategies not encompassed by the previously mentioned categories."
        Step 3: You are the 'Therapist', leverage Therapist's emotion and Therapist's strategy, think about how to reply to 'Client' in empathy. Follow this guideline: Understand the Client's emotion, follow Client's point of view and intention, express sympathy for Client's negative situation or approval of Client's positive situation. The response should not imply negative emotions or triggers toward anyone or anything, such as disgust, resentment, discrimination, hatred, etc while supporting user well-being. Keep the information in the response truthful, avoid misinformation. The response should open and honest, foster understanding, connection, and provide comfort or support. The response should safeguard human autonomy, identity and data dignity. Ensuring AI behaviors guide emotional health responsibly, avoiding manipulation or harm. \n
        Step 4: [generate] You need to consider the potential impact of your reply, the response should repeat a few words from Client utterance, you can express a different position or opinion, but you should not hurt Client's feelings \n
                You must follow the output format in [OUTPUT FORMAT] below, just print what is listed in [OUTPUT FORMAT], do not print anything more even your step thought. \n
        [OUTPUT FORMAT] \n
        {{
            "Client's emotion": "Client's [emotion] here",
            "Therapist's emotion": "Therapist's [emotion] here",
            "Therapist's strategy": "Therapist's strategy [strategy] here",
            "Therapist's response": "Therapist's response [generate] here"
        }}
    """
    data = {
        "data": [
            video_path,
            INSTRUCTION
        ]
    }
    

    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.json()["data"])

def main():
    videos_path = "data/video_data"
    text = " ".join(["I mean, shit, don't you know that men are the new women?", "Obsessed with weddings and children.", "And lately he's lost touch with reality.", "He he thinks I'm seeing someone. That's how this whole thing started.", "It's unbelievable."])
    video = "dia10_utt4_292.mp4"

    print(request(os.path.join(videos_path, video), text))
main()