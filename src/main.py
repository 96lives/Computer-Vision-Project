from finger_counter_tester import FingerCounterTester

if __name__ == "__main__":
    
    in_dir = "../test-data/"
    video_name = "IMG_0085.MOV"
    out_dir = "../test-data/out/"
    report_name = "report.txt"
    fc = FingerCounterTester(video_name, report_name, in_dir, out_dir, True)
    fc.play_game()   
