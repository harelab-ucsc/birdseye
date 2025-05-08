import glob2
import os
from PIL import Image
import imageio.v2 as imageio  # make sure imageio is installed


def make_gif(frame_folder, gif_name, pattern="2d_*.png"):
	file_list = glob2.glob(os.path.join(frame_folder, pattern))
	file_list.sort()
	print(f"Found {len(file_list)} frames.")

	# Load images into memory and close file handles
	frames = []
	for i, image_path in enumerate(file_list):
	    with Image.open(image_path) as img:
	        frames.append(img.copy())

	print(f"Loaded {len(frames)} frames into memory.")

	frame_one = frames[0]
	frame_one.save(f"{gif_name}.gif", format="GIF", append_images=frames[1:], save_all=True,
	               duration=200, loop=0)
	print(f"Saved GIF as {gif_name}.mp4")


def make_mp4(frame_folder, video_name, fps=5, pattern="2d_*.png"):
	file_list = glob2.glob(os.path.join(frame_folder, pattern))
	file_list.sort()
	print(f"Found {len(file_list)} frames.")
	save_path = os.path.join(frame_folder, f"{video_name}.mp4")
	with imageio.get_writer(save_path, fps=fps, codec='libx264', format='ffmpeg') as writer:
	    for image_path in file_list:
	        img = imageio.imread(image_path)
	        writer.append_data(img)
	print(f"Saved video as {save_path}")

if __name__ == "__main__":
	mode = 'mp4'
	filepath = os.path.expanduser('~')
	filepath = os.path.join(filepath, 'catch', 'tmp')
	print(filepath)
	if mode == 'mp4':
		make_mp4(filepath, 'flight_video_2d', fps=5)
	elif mode == 'gif':
		make_gif(filepath, 'flight_gif')
