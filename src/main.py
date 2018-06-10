from new_finger_counter import NewFingerCounter as NFC
from finger_counter import FingerCounter as FC

if __name__ == "__main__":
    
    in_dir = "../data/hardR/"
    video_name = "IMG_0048.MOV"
    out_dir = "../data/out/"
    report_name = "report.txt"
    fc = NFC(video_name, report_name, in_dir, out_dir, True)
    fc.play_game()   
