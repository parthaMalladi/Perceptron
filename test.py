import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import numpy as np
from hopfieldNN import HopfieldNN

CANVAS_SIZE = 280  # Canvas in pixels
RESIZE_SIZE = 28   # Target Hopfield input size

class DigitDrawer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Draw a digit")
        self.mem = []
        
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.draw)
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        
        tk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Get Vector", command=self.get_vector).pack(side=tk.LEFT)
        
        self.last_x, self.last_y = None, None
        self.root.mainloop()
    
    def draw(self, event):
        r = 8  # brush radius
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # Draw a line from last position to current
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=r*2, fill='black', capstyle=tk.ROUND, smooth=True)
        self.last_x, self.last_y = x, y
    
    def clear(self):
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None
    
    def get_vector(self):
        # Capture the canvas
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + CANVAS_SIZE
        y1 = y + CANVAS_SIZE
        img = ImageGrab.grab(bbox=(x, y, x1, y1))
        
        # Convert to grayscale and invert (so black=0, white=255)
        img = ImageOps.grayscale(img)
        img = ImageOps.invert(img)
        
        # Resize to 28x28
        img = img.resize((RESIZE_SIZE, RESIZE_SIZE), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        arr = np.array(img)
        
        # Threshold: black pixels → -1, white → +1
        vector = np.where(arr > 128, 1.0, -1.0)
        
        # Flatten to 1D
        vector = vector.flatten().tolist()
        
        # store the vector in memory
        self.mem.append(vector)
        print("Digit Added")
        
        # run hopfield network. First drawing is memory, second drawing is perturbed
        size = len(self.mem)
        if size >= 3:
            # initialize Hopfield network
            network = HopfieldNN(784)
            
            # store each memory and train
            for i in range(size - 1):
                network.storeMemory(self.mem[i])
            network.train()
            
            # classify a perturbed image
            recall = network.classify(self.mem[size - 1])
            
            # see what was recalled
            for i in range(size - 1):
                if recall == self.mem[i]:
                    print("Recalled to index", i)

# Run the drawer
DigitDrawer()