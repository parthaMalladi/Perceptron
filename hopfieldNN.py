class HopfieldNN:
    def __init__(self, n=0):
        # number of neurons in the hopfield network
        self.neurons = n
        
        # to keep track of the weight matrix
        self.matrix = []
        
        # to keep track of the memories
        self.memories = []
        
        # create a n x n weight matrix
        for _ in range(self.neurons):
            row = [0] * self.neurons
            self.matrix.append(row)
    
    # store the memories for later use
    def storeMemory(self, mem):
        self.memories.append(mem)
    
    # train the network by creating the correct weight matrix
    def train(self):
        for mem in self.memories:
            # each weight is the dot product of the memories with itself
            for i in range(self.neurons):
                for j in range(self.neurons):
                    if j == i:
                        continue
                    self.matrix[i][j] += (mem[i] * mem[j])/self.neurons
    
    # recall some memory based on perturbed input
    def classify(self, input):
        while True:
            isUpdated = False
            
            # iterate through each nueron to see if it needs to be flipped
            for i in range(len(input)):
                # sum to see the output of neuron i
                sum = 0
                
                # go to the row in the weight matrix corresponding to neuron i
                for w in range(len(self.matrix[i])):
                    if w == i:
                        continue
                    
                    # find the weighted sum
                    sum += self.matrix[i][w] * input[w]
                
                # flip the output if necessary
                if sum > 0.0:
                    if input[i] == -1.0:
                        isUpdated = True
                        input[i] = 1.0
                else:
                    if input[i] == 1.0:
                        isUpdated = True
                        input[i] = -1.0
            
            # break out of the loop if no neuron got updated
            if not isUpdated:
                break
        
        # return the recalled memory
        return input
                