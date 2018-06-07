from new_finger_counter import NewFingerCounter as FC

if __name__ == "__main__":
    
    in_dir = "../data/hardP/"
    video_name = "IMG_0050.MOV"
    out_dir = "../data/out/"
    report_name = "report.txt"
    fc = FC(video_name, report_name, in_dir, out_dir, True)
    fc.play_game()   
