from webcam_finger_counter import WebFingerCounter as WFC

report = "report.txt"
out_dir = "../data/out/"

fc = WFC(report, out_dir, True)
fc.play_game()
