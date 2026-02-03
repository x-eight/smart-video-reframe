from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
import sys, os, pickle

class SceneDetector:
    def __init__(self, videoFilePath, pyworkPath):
        # CPU: Scene detection, output is the list of each shot's time duration
        self.videoFilePath = videoFilePath
        self.pyworkPath = pyworkPath
        self.videoManager = VideoManager([videoFilePath])
        self.statsManager = StatsManager()
        self.sceneManager = SceneManager(self.statsManager)
        self.sceneManager.add_detector(ContentDetector())
        self.baseTimecode = self.videoManager.get_base_timecode()
        self.videoManager.set_downscale_factor()
        self.videoManager.start()
        self.sceneManager.detect_scenes(frame_source = self.videoManager)

    def get_frame_by_scene(self):
        sceneList = self.sceneManager.get_scene_list(self.baseTimecode)

        savePath = os.path.join(self.pyworkPath, 'scene.pckl')
        if sceneList == []:
            sceneList = [(self.videoManager.get_base_timecode(),self.videoManager.get_current_timecode())]
        with open(savePath, 'wb') as fil:
            pickle.dump(sceneList, fil)
            sys.stderr.write('%s - scenes detected %d\n'%(self.videoFilePath, len(sceneList)))
        #return sceneList
        frames = []
        for start, end in sceneList:
            frames.append({
                "start_seconds": start.get_seconds(),
                "start_frame": start.frame_num,
                "end_seconds": end.get_seconds(),
                "end_frame": end.frame_num,
            })
 
        return frames
