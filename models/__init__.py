
from models.coreContinuous_IGEV.continuous_IGEVstereo import continuous_IGEVStereo
from models.corePrune_RAFT.prune_raft_stereo import continuous_RaftStereo
__models__ = {
    "continuous_IGEVStereo": continuous_IGEVStereo,
    "continuous_RAFTStereo": continuous_RaftStereo
}
