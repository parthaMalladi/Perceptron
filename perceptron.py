class Perceptron:
    def __init__(self):
        self.weights = []
        self.dimensions = 0
    
    # finds the dot product between any two vectors
    def dotProd(self, x, y):
        ans = 0
        for i in range(len(x)):
            ans += (x[i] * y[i])
        
        return ans

    # preps the training samples to work with the perceptron weight vector
    def prepSamples(self, samples):
        return [(input + [1], label) for input, label in samples]
        
    def train(self, samples):
        # get the dimensions of the samples
        self.dimensions = len(samples[0][0])
        
        # create a weight vector with the bias term and initialize it to w = 0
        self.weights = [0] * (self.dimensions + 1)
        
        # prep samples so that they can work with the bias term
        samples = self.prepSamples(samples)
        
        while True:
            # for each sample, check if the weight vector needs to be updated
            isUpdated = False
            
            for input, label in samples:
                check = self.dotProd(self.weights, input)
                if label * check <= 0:
                    isUpdated = True
                    # update w so that it gets closer to a hyperplane
                    for i in range(len(input)):
                        updateStep = label * input[i]
                        self.weights[i] += updateStep
            
            # break out of the loop if a seperating hyperplane is found
            if isUpdated == False:
                break
        
    # classify a new, unlabeled data point    
    def classify(self, input):
        res = self.dotProd(input + [1], self.weights)
        
        if res > 0:
            return 1
        else:
            return -1