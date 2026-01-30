from hopfieldNN import HopfieldNN

network = HopfieldNN(3)
network.storeMemory([-1.0,1.0,-1.0])
network.train()
network.classify([1.0,1.0,-1.0])