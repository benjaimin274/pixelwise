import os  
import tkinter as tk  
from tkinter import messagebox  
from PIL import Image, ImageDraw

# --- Global variables ---  
SAVE_DIR = "drawings class B"
CANVAS_PIXELS = 28 #MNISt size
SCALE = 10 # Display size 
DISPLAY_SIZE = CANVAS_PIXELS * SCALE  
BRUSH_RADIUS = 10  # Brush radius on the large canvas (~2px at 28x28)

os.makedirs(SAVE_DIR, exist_ok=True)


class DrawingApp:  
    def __init__(self, root: str, label: str = None):  
        self.root = root
        self.label = label  
        self.root.title("MNIST-Style Drawing App (28x28)")

        # Black background  
        self.hires_image = Image.new("L", (DISPLAY_SIZE, DISPLAY_SIZE), color=0)  
        self.draw = ImageDraw.Draw(self.hires_image)

        # Tkinter canvas for visual drawing  
        self.canvas = tk.Canvas(  
            root,  
            width=DISPLAY_SIZE,  
            height=DISPLAY_SIZE,  
            bg="black",  
            cursor="pencil",  
        )  
        self.canvas.pack(padx=10, pady=10)  
        self.canvas.bind("<Button-1>", self.start_paint)  
        self.canvas.bind("<B1-Motion>", self.paint)

        self.last_x = None  
        self.last_y = None

        # Controls  
        controls = tk.Frame(root)  
        controls.pack(pady=5)

        tk.Label(controls, text="Label:").pack(side=tk.LEFT)  
        self.label_entry = tk.Entry(controls, width=20)  
        self.label_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(controls, text="Save", command=self.save_image,  
                  width=10).pack(side=tk.LEFT, padx=5)  
        tk.Button(controls, text="Clear", command=self.clear_canvas,  
                  width=10).pack(side=tk.LEFT, padx=5)

    def start_paint(self, event):  
        self.last_x, self.last_y = event.x, event.y  
        self._draw_brush(event.x, event.y)

    def paint(self, event):  
        x, y = event.x, event.y  
        if self.last_x is not None and self.last_y is not None:  
            # Draw smooth line on visual canvas  
            self.canvas.create_line(  
                self.last_x, self.last_y, x, y,  
                fill="white",  
                width=BRUSH_RADIUS * 2,  
                capstyle=tk.ROUND,  
                smooth=True,  
            )  
            # Draw matching line on hi-res PIL image  
            self.draw.line(  
                [(self.last_x, self.last_y), (x, y)],  
                fill=255, # white pencil 
                width=BRUSH_RADIUS * 2,  
            )  
            # Round caps via circles at both ends  
            self._draw_brush(x, y)

        self.last_x, self.last_y = x, y

    def _draw_brush(self, x, y):  
        # Round brush dot on both Tk canvas and PIL image  
        r = BRUSH_RADIUS  
        self.canvas.create_oval(x - r, y - r, x + r, y + r,  
                                fill="white", outline="white")  
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear_canvas(self):  
        self.canvas.delete("all")  
        self.hires_image = Image.new("L", (DISPLAY_SIZE, DISPLAY_SIZE), color=0)  
        self.draw = ImageDraw.Draw(self.hires_image)  
        self.last_x = None  
        self.last_y = None

    def save_image(self):
        if self.label is None:  
            self.label = self.label_entry.get().strip()  
        if not self.label:
            # No label was selected  
            messagebox.showwarning("Missing label", "Please enter a label before saving.")  
            return

        # Sanitize label for filename  
        safe_label = "".join(c for c in self.label if c.isalnum() or c in ("_", "-")).strip()  
        if not safe_label:  
            messagebox.showwarning("Invalid label", "Label contains no valid characters.")  
            return

        # Downscale to 28 x 28 pixels 
        final_image = self.hires_image.resize(  
            (CANVAS_PIXELS, CANVAS_PIXELS),  
            resample=Image.LANCZOS,  
        )

        # Adjust filename (label_1.png, label_2.png, ...)  
        counter = 1  
        while True:  
            filename = f"{safe_label}_{counter}.png"  
            filepath = os.path.join(SAVE_DIR, filename)  
            if not os.path.exists(filepath):  
                break  
            counter += 1

        final_image.save(filepath)   
        self.clear_canvas()  
        self.label_entry.delete(0, tk.END)


if __name__ == "__main__":  
    root = tk.Tk()  
    app = DrawingApp(root, "nine")  
    root.mainloop()  
