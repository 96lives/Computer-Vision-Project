CHEATING ROCK SCISSOR PAPER 

File Structure

1. finger_counter.py
    Contains class "FingerCounter", which is the core class of the project.
    This class counts and predicts the fingerCounter

2. skin_detection.py
    Contains functions for detecting skin

3. background_subtractor.py
    Contains class "BackgroundSubtractor", which subtracts the background to detect hand.
    Since our dataset consists videos that dont have pure background, this method is only used for testing

4. shaker.py 
    Contains class "Shaker", which detects the 2 updown movement of the RPS game

5. classifier.py
    Contains class "SkinColorClassifier", which trains the skin color by comparing 2 hand images on intialization.
    During test time, it inputs image and classifies returns the skin mask with respect to trained data.
    Uses either random forest classifier or k nearest neighbors classifier.

6. rf_classifier.py 
    Contains class "RFClassifier", which is a wrapper for sklearn RandomForestClassifier

7. knn_classifier.py 
    Contains class "KNNClassifier", which is a wrapper for sklearn KNClassifier 


