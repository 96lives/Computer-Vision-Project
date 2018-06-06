from finger_counter2 import FingerCounter2
from finger_counter3 import FingerCounter3

if __name__ == "__main__":
    
    in_dir = "../test-data/"
    video_name = "IMG_0112.MOV"
    out_dir = "../test-data/out/"
    report_name = "report.txt"
    fc = FingerCounter3(video_name, report_name, in_dir, out_dir, True)
    fc.play_game()   
