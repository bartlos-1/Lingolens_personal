import tensorflow as tf
from typing import List
import cv2
import os 
from difflib import SequenceMatcher
# Define the vocabulary for character mapping
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Create a mapping from characters to integers
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# Create a mapping from integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_data(path: str): 
    # Convert the path from bytes to string
    path = bytes.decode(path.numpy())
    
    # Extract the file name from the path
    file_name = path.split('/')[-1].split('.')[0]
    
    # IF RUNNNING ON WINDOWS UNCOMMENT LINE BELOW, AND PUT THE LINE ABOVE IN COMMENTS
    # file_name = path.split('\\')[-1].split('.')[0]
    
    # Construct the paths to the video and alignment files
    video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
    
    # Load the video frames and alignments
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments

def load_alignments(path:str) -> List[str]: 
    # Read the lines from the alignment file
    with open(path, 'r') as f: 
        lines = f.readlines() 
    
    # Extract the tokens from the lines, excluding 'sil' tokens
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    
    # Convert the tokens to integers using the char_to_num mapping
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_video(path:str) -> List[float]: 
    # Open the video file
    cap = cv2.VideoCapture(path)
    frames = []
    # Iterate over each frame in the video
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        # Convert the frame to grayscale
        frame = tf.image.rgb_to_grayscale(frame)
        # Extract a region of interest from the frame
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    # Calculate the mean and standard deviation of the frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    
    # Normalize the frames using the mean and standard deviation
    return tf.cast((frames - mean), tf.float32) / std


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()