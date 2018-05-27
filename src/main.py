import fingerCounter

if __name__ == "__main__":
    #fc = fingerCounter.FingerCounter('skin')
    fc = fingerCounter.FingerCounter('skin', in_dir = '../../../data/smallR/2.MOV')
    fc.play_game()    
