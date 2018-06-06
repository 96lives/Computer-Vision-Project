from finger_counter import FingerCounter

if __name__ == "__main__":
    in_dir = "../data/smallP/IMG_0131.MOV"
    out_dir = "../test-data/out.avi"
    fc = FingerCounter('skin', in_dir, out_dir)
    #fc = FingerCounter('background')
    fc.play_game()    
