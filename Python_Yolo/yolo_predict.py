from ultralytics import YOLO
import os
import time
import psutil

# Get the process ID (PID) of the current process
pid = psutil.Process().pid

# Create a Process object
process = psutil.Process(pid)
process.cpu_percent()

model = YOLO("yolov8s.pt")

start_time = time.time()

# Define path to the image file
source_dir = "../zips/val2017/"

no_of_imgs = 0
max_possible_images = len(os.listdir(source_dir))
if no_of_imgs > 0:
    no_of_imgs = min(no_of_imgs, max_possible_images)
else:
    no_of_imgs = max_possible_images

print(f"Calculating Stats for {no_of_imgs} images")
'''
youtube_links = []
with open('youtube_links.txt', 'r') as file:
    for line in file:
        youtube_links.append(line.strip())
'''
# Read the entire directory
results = model(source_dir)

# for source in os.listdir(source_dir)[:no_of_imgs]:
    # model(f"/home/aadityapal/Work/Neophyte/cmp_mojo_python/zips/val2017/{source}")

# results = [model(link, stream=True) for link in youtube_links]
elapsed_time = time.time() - start_time
# fps = no_of_imgs/elapsed_time
fps = len(results)/elapsed_time
avg_cpu_usage = process.cpu_percent()/psutil.cpu_count()
print(f"Average Cpu usage: {avg_cpu_usage}%")
# print(f"CPU Usage: {sum(cpu_usage_data)/len(cpu_usage_data)}")
print(f"Elapsed time: {elapsed_time} seconds")
print(f"FPS: {fps:.2f}")

with open("results.txt", "a") as f:
    f.write("##############################\n")
    f.write("RUN STATS\n")
    f.write("##############################\n")

    f.write(f"Total Frames:{no_of_imgs}\n")
    f.write(f"Average Cpu usage: {avg_cpu_usage}%\n")
    f.write(f"Elapsed time: {elapsed_time} seconds\n")
    f.write(f"FPS: {fps:.2f}\n")
    f.write("\n\n")
