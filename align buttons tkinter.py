# Chỉ dùng grid hoặc pack, không thể kết hợp
# Align với grid

import tkinter as tk

root = tk.Tk()

button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, side=tk.BOTTOM)

reset_button = tk.Button(button_frame, text='Reset')
run_button = tk.Button(button_frame, text='Run')
t_button = tk.Button(button_frame, text='1111')
r_button = tk.Button(button_frame, text='1')
e_button = tk.Button(button_frame, text='1')

button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)
button_frame.columnconfigure(2, weight=1)

reset_button.grid(row=0, column=0, sticky=tk.W + tk.E)
run_button.grid(row=0, column=1, sticky=tk.W + tk.E)
t_button.grid(row=1, column=2, sticky=tk.W + tk.E)
r_button.grid(row=1, column=1, sticky=tk.W + tk.E)

root.mainloop()

# Align với pack
from tkinter import LEFT, BOTTOM, Frame, Tk, Button

root = Tk()
frame = Frame(root)
frame.pack()

bottomframe = Frame(root)
bottomframe.pack(side=BOTTOM)

redbutton = Button(frame, text="Red", bg="red")
redbutton.pack(side=LEFT)

greenbutton = Button(frame, text="green", fg="green")
greenbutton.pack(side=LEFT)

bluebutton = Button(frame, text="Blue", fg="blue")
bluebutton.pack(side=LEFT)

blackbutton = Button(bottomframe, text="Black", fg="black")
blackbutton.pack(side=BOTTOM)

root.mainloop()
