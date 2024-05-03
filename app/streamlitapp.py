# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from utils import load_data, num_to_char, similar
from modelutil import load_model



# Set the title and description of the Streamlit app
st.title('Lingolens') 
st.markdown('### CPSC 571 Project')

# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options, index=None, placeholder="Select video...")



# Check if the selected video is in the list of options
if selected_video in options: 

    # Rendering the video 
    with st.container(border =1): 
        # Construct the file path of the selected video
        file_path = os.path.join('..','data','s1', selected_video)
        
        # Convert the video to a format that can be displayed in the app
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Display the video in the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)
        
        # Add a button to trigger the lip-reading process
        clicked = st.button('Lip read', use_container_width=True)
    
    # Check if the "Lip read" button was clicked
    if (clicked):
        with st.container(): 
            # Load the video and its annotations
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            real_text = tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(annotations.numpy()).numpy()]).numpy().decode("utf-8")
            st.info('Real text:')
            st.text(real_text)
            # Display a message indicating what the model sees
            st.info('What the model sees:')
            
            # Create an animation from the video frames
            imageio.mimsave('animation.gif', video, fps=10)
            st.image('animation.gif', width=400) 
            
            # Display a message indicating that the lip-reading process is in progress
            st.info('Token output of the model:')
            
            # Perform lip reading using the loaded model
            with st.spinner('Lip reading...'):
                model = load_model()
                yhat = model.predict(tf.expand_dims(video, axis=0))
                decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
                st.text(decoder)
                
                
            # Convert the model's prediction to text
            st.success('Decoded text:')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)

            accuracy = similar(real_text, converted_prediction)

            st.info('Accuracy:')
            st.text(accuracy)
              