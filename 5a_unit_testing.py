"""
Unit tests for the emotion_detector function.
Uses unittest.mock to simulate Watson NLP API responses.
"""
import unittest
from unittest.mock import patch, MagicMock
import json
from EmotionDetection.emotion_detection import emotion_detector


def make_mock_response(anger, disgust, fear, joy, sadness):
    """Helper to build a mock requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = json.dumps({
        "emotionPredictions": [{
            "emotion": {
                "anger": anger,
                "disgust": disgust,
                "fear": fear,
                "joy": joy,
                "sadness": sadness
            }
        }]
    })
    return mock_resp


class TestEmotionDetector(unittest.TestCase):
    """Test cases for the emotion_detector function."""

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_joy_emotion(self, mock_post):
        """Test that 'I am glad this happened' returns joy as dominant emotion."""
        mock_post.return_value = make_mock_response(0.1, 0.05, 0.05, 0.8, 0.1)
        result = emotion_detector('I am glad this happened')
        self.assertEqual(result['dominant_emotion'], 'joy')

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_anger_emotion(self, mock_post):
        """Test that 'I am really mad about this' returns anger as dominant emotion."""
        mock_post.return_value = make_mock_response(0.8, 0.1, 0.05, 0.05, 0.1)
        result = emotion_detector('I am really mad about this')
        self.assertEqual(result['dominant_emotion'], 'anger')

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_disgust_emotion(self, mock_post):
        """Test that 'I feel disgusted just hearing about this' returns disgust."""
        mock_post.return_value = make_mock_response(0.1, 0.8, 0.05, 0.05, 0.1)
        result = emotion_detector('I feel disgusted just hearing about this')
        self.assertEqual(result['dominant_emotion'], 'disgust')

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_sadness_emotion(self, mock_post):
        """Test that 'I am so sad about this' returns sadness as dominant emotion."""
        mock_post.return_value = make_mock_response(0.1, 0.05, 0.05, 0.1, 0.8)
        result = emotion_detector('I am so sad about this')
        self.assertEqual(result['dominant_emotion'], 'sadness')

    @patch('EmotionDetection.emotion_detection.requests.post')
    def test_fear_emotion(self, mock_post):
        """Test that 'I am really afraid that this will happen' returns fear."""
        mock_post.return_value = make_mock_response(0.1, 0.05, 0.8, 0.05, 0.1)
        result = emotion_detector('I am really afraid that this will happen')
        self.assertEqual(result['dominant_emotion'], 'fear')


if __name__ == '__main__':
    unittest.main()
