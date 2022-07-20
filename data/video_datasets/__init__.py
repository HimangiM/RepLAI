from .ucf101 import UCF101
from .hmdb51 import HMDB51
from .epic_kitchens import EpicKitchens, EpicKitchensSegments, EpicKitchensActionSegments, EpicKitchensV2, \
    EpicKitchensAudioPeakSegments, EpicKitchensActionAudioPeakSegments, EpicKitchensAudioPeakSegments55, \
    EpicKitchensActionSegments55, EpicKitchensActionSegmentsUnseenParticipant, EpicKitchensActionSegmentsTailClasses, \
    EpicKitchensActionSegmentsHeadClasses
from .ava import AVA
from .kinetics import Kinetics

import av
av.logging.set_level(av.logging.ERROR)