"""
Emotion Detection module using Watson NLP Library.
This module provides the emotion_detector function that calls the
Watson NLP Emotion Predict API and returns emotion scores.
"""

import json
import requests


def emotion_detector(text_to_analyse):
    """
    Detect emotions in the given text using Watson NLP Emotion Predict API.

    Args:
        text_to_analyse (str): The text to analyse for emotions.

    Returns:
        dict: A dictionary containing emotion scores and the dominant emotion.
              Returns None values if the input is blank or status code is 400.
    """
    # Handle blank input
    if text_to_analyse is None or text_to_analyse.strip() == "":
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    url = (
        'https://sn-watson-emotion.labs.skills.network'
        '/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    myobj = {
        "raw_document": {
            "text": text_to_analyse
        }
    }

    response = requests.post(url, json=myobj, headers=headers)

    # Handle error status code 400
    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    # Parse the response
    response_dict = json.loads(response.text)

    # Extract emotion predictions
    emotions = response_dict['emotionPredictions'][0]['emotion']

    anger_score = emotions['anger']
    disgust_score = emotions['disgust']
    fear_score = emotions['fear']
    joy_score = emotions['joy']
    sadness_score = emotions['sadness']

    # Find dominant emotion
    emotion_scores = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score
    }

    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }

