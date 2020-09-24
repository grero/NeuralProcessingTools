import NeuralProcessingTools as NPT


def test_spikebursts():
    spiketrain = [0.1, 0.3, 0.4, 0.45, 0.46, 0.47, 0.7, 0.8, 0.81, 0.83, 0.85, 0.91]
    burst_idx, burst_length = NPT.bursts.find_spikebursts(spiketrain)
    assert burst_idx == [3, 7]
    assert burst_length == [3, 2]
    

